# -*- coding: utf-8 -*-
"""
Fix Heuristics for GL-FOUND-X-002.

This module implements heuristics for generating fix suggestions based on
validation findings. Each heuristic addresses a specific type of error and
generates appropriate JSON Patch operations with safety classification.

Heuristics per PRD Appendix C:
    1. Rename field when schema provides `renamed_from`
    2. Add optional defaults when default exists
    3. Coerce safe primitives with exact parsing
    4. Unit conversion when dimension matches
    5. Close-match unknown keys (edit distance <= 2)

Example:
    >>> from greenlang.schema.suggestions.heuristics import FixHeuristics
    >>> from greenlang.schema.compiler.ir import SchemaIR
    >>> heuristics = FixHeuristics(ir)
    >>> suggestion = heuristics.suggest_for_finding(finding, payload)
    >>> if suggestion:
    ...     print(f"[{suggestion.safety}] {suggestion.rationale}")

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 4.3
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Literal

from ..compiler.ir import SchemaIR, PropertyIR, UnitSpecIR
from ..models.finding import Finding, Severity
from ..models.patch import FixSuggestion, PatchSafety, JSONPatchOp as ModelJSONPatchOp
from .patches import PatchGenerator, JSONPatchOperation, PatchOp
from .safety import (
    PatchSafetyClassifier,
    PatchContext,
    SafetyClassification,
    JSONPatchOperation as SafetyJSONPatchOperation,
)


logger = logging.getLogger(__name__)


# =============================================================================
# HEURISTIC CONSTANTS
# =============================================================================

MAX_TYPO_DISTANCE: int = 2
"""Maximum Levenshtein edit distance for typo detection."""

MAX_ENUM_DISTANCE: int = 2
"""Maximum edit distance for enum value suggestion."""

RANGE_TYPO_THRESHOLD: float = 0.1
"""Threshold for range violation typo detection (10% of range)."""

BOOLEAN_TRUE_VALUES: Set[str] = {"true", "yes", "1", "on"}
"""String values that can be coerced to True."""

BOOLEAN_FALSE_VALUES: Set[str] = {"false", "no", "0", "off"}
"""String values that can be coerced to False."""


# =============================================================================
# FIX HEURISTICS CLASS
# =============================================================================


class FixHeuristics:
    """
    Generates fix suggestions using heuristics.

    This class implements various heuristics to generate fix suggestions
    for validation findings. Each heuristic is designed for a specific
    error type and applies safety classification to ensure patches
    don't inadvertently corrupt data.

    Attributes:
        ir: Compiled schema Intermediate Representation
        _generator: JSON Patch generator
        _classifier: Patch safety classifier

    Heuristics:
        1. renamed_from: Schema-declared field renames
        2. defaults: Optional field defaults from schema
        3. coercion: Safe primitive type coercion
        4. unit_conversion: Same-dimension unit conversion
        5. typo_correction: Edit distance-based field matching

    Example:
        >>> from greenlang.schema.compiler.ir import SchemaIR
        >>> ir = SchemaIR(...)
        >>> heuristics = FixHeuristics(ir)
        >>> suggestion = heuristics.suggest_for_finding(finding, payload)
    """

    def __init__(self, ir: SchemaIR):
        """
        Initialize FixHeuristics.

        Args:
            ir: Compiled schema IR for reference
        """
        self.ir = ir
        self._generator = PatchGenerator()
        self._classifier = PatchSafetyClassifier(ir)

    # -------------------------------------------------------------------------
    # Main Dispatch Method
    # -------------------------------------------------------------------------

    def suggest_for_finding(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Generate fix suggestion for a finding.

        Dispatches to appropriate heuristic based on error code.

        Args:
            finding: The validation finding to generate a fix for
            payload: The original payload being validated

        Returns:
            FixSuggestion if a fix can be generated, None otherwise

        Example:
            >>> finding = Finding(code="GLSCHEMA-E100", ...)
            >>> suggestion = heuristics.suggest_for_finding(finding, payload)
        """
        # Dispatch based on error code
        dispatch_map = {
            "GLSCHEMA-E100": self.suggest_for_missing_required,
            "GLSCHEMA-E101": self.suggest_for_unknown_field,
            "GLSCHEMA-E102": self.suggest_for_type_mismatch,
            "GLSCHEMA-E200": self.suggest_for_range_violation,
            "GLSCHEMA-E202": self.suggest_for_enum_violation,
            "GLSCHEMA-E301": self.suggest_for_unit_incompatible,
            "GLSCHEMA-E302": self.suggest_for_unit_noncanonical,
            "GLSCHEMA-W600": self.suggest_for_deprecated_field,
            "GLSCHEMA-W601": self.suggest_for_renamed_field,
        }

        handler = dispatch_map.get(finding.code)
        if handler is None:
            logger.debug(
                f"No heuristic available for error code {finding.code}"
            )
            return None

        try:
            return handler(finding, payload)
        except Exception as e:
            logger.warning(
                f"Heuristic failed for {finding.code} at {finding.path}: {e}"
            )
            return None

    # -------------------------------------------------------------------------
    # Structural Error Heuristics (E1xx)
    # -------------------------------------------------------------------------

    def suggest_for_missing_required(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E100 (missing required).

        Only suggests if schema has default value.
        Safety: safe if optional with default, unsafe if truly required.

        Args:
            finding: The missing required field finding
            payload: Original payload

        Returns:
            FixSuggestion if default exists, None otherwise
        """
        path = finding.path
        prop = self.ir.get_property(path)

        if prop is None:
            logger.debug(f"No property IR found for path {path}")
            return None

        # Only suggest if there's a default value
        if not prop.has_default:
            logger.debug(f"No default value for required field {path}")
            return None

        default_value = prop.default_value

        # Generate add patch
        patches = [self._generator.generate_add(path, default_value)]

        # Determine safety based on whether field is truly required
        if prop.required:
            # Field is required and we're using a default - needs review
            safety = PatchSafety.NEEDS_REVIEW
            rationale = (
                f"Adding required field with schema default value: "
                f"{repr(default_value)}"
            )
            confidence = 0.7
        else:
            # Field is optional with default - safe
            safety = PatchSafety.SAFE
            rationale = (
                f"Adding optional field with schema default value: "
                f"{repr(default_value)}"
            )
            confidence = 0.95

        context = PatchContext(
            finding=finding,
            original_value=None,
            suggested_value=default_value,
            derivation="schema_default",
            schema_has_default=True,
            is_required_field=prop.required
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    def suggest_for_type_mismatch(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E102 (type mismatch).

        Attempts safe coercion:
        - "42" -> 42 for integer
        - "3.14" -> 3.14 for number
        - "true"/"false" -> True/False for boolean

        Args:
            finding: The type mismatch finding
            payload: Original payload

        Returns:
            FixSuggestion if safe coercion possible, None otherwise
        """
        path = finding.path
        actual_value = self._get_value_at_path(payload, path)

        if actual_value is None:
            return None

        # Get expected type from finding
        expected = finding.expected or {}
        expected_type = expected.get("type")

        if not expected_type:
            # Try to get from IR
            prop = self.ir.get_property(path)
            if prop and prop.type:
                expected_type = prop.type
            else:
                return None

        # Attempt coercion
        can_coerce, coerced_value = self._can_coerce(actual_value, expected_type)

        if not can_coerce:
            logger.debug(
                f"Cannot safely coerce {type(actual_value).__name__} to {expected_type}"
            )
            return None

        # Generate replace patch with test precondition
        patches = self._generator.generate_replace(
            path=path,
            old_value=actual_value,
            new_value=coerced_value
        )

        # Exact coercion is safe
        safety = PatchSafety.SAFE
        rationale = (
            f"Coercing {type(actual_value).__name__} "
            f"'{actual_value}' to {expected_type} {repr(coerced_value)}"
        )
        confidence = 0.95

        context = PatchContext(
            finding=finding,
            original_value=actual_value,
            suggested_value=coerced_value,
            derivation="exact_coercion",
            is_type_coercion=True
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    def suggest_for_unknown_field(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E101 (unknown field).

        Checks for:
        1. Close match to known field (typo)
        2. Renamed field (schema declares renamed_from)

        Safety: needs_review for typo, safe for declared rename.

        Args:
            finding: The unknown field finding
            payload: Original payload

        Returns:
            FixSuggestion if match found, None otherwise
        """
        path = finding.path
        unknown_key = self._get_key_from_path(path)

        if not unknown_key:
            return None

        # Get known keys from IR
        known_keys = self._get_known_keys_for_parent(path)

        # Check 1: Is this a declared rename (renamed_from)?
        for old_key, new_key in self.ir.renamed_fields.items():
            if unknown_key == old_key:
                # This is a declared rename - safe
                old_path = path
                new_path = self._replace_last_key(path, new_key)

                patches = self._generator.generate_field_rename(old_path, new_path)

                safety = PatchSafety.SAFE
                rationale = (
                    f"Field '{unknown_key}' has been renamed to '{new_key}' "
                    f"per schema definition"
                )
                confidence = 1.0

                context = PatchContext(
                    finding=finding,
                    original_value=None,
                    suggested_value=None,
                    derivation="schema_rename",
                    is_alias_resolution=True
                )

                return self._build_suggestion(
                    patches=patches,
                    context=context,
                    finding=finding,
                    safety_override=safety,
                    confidence=confidence,
                    rationale=rationale
                )

        # Check 2: Is this a typo? (edit distance <= MAX_TYPO_DISTANCE)
        close_matches = self._find_close_matches(
            key=unknown_key,
            known_keys=known_keys,
            max_distance=MAX_TYPO_DISTANCE
        )

        if close_matches:
            best_match, distance = close_matches[0]

            old_path = path
            new_path = self._replace_last_key(path, best_match)

            # Check if target field already exists
            target_value = self._get_value_at_path(payload, new_path)
            if target_value is not None:
                logger.debug(
                    f"Target field '{best_match}' already exists, cannot rename"
                )
                return None

            patches = self._generator.generate_field_rename(old_path, new_path)

            # Typo correction needs review
            safety = PatchSafety.NEEDS_REVIEW
            rationale = (
                f"Possible typo: '{unknown_key}' -> '{best_match}' "
                f"(edit distance: {distance})"
            )
            # Confidence decreases with distance
            confidence = 0.9 - (distance * 0.1)

            context = PatchContext(
                finding=finding,
                original_value=None,
                suggested_value=None,
                derivation="typo_correction",
                edit_distance=distance
            )

            return self._build_suggestion(
                patches=patches,
                context=context,
                finding=finding,
                safety_override=safety,
                confidence=confidence,
                rationale=rationale
            )

        return None

    # -------------------------------------------------------------------------
    # Constraint Error Heuristics (E2xx)
    # -------------------------------------------------------------------------

    def suggest_for_range_violation(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E200 (range violation).

        Generally unsafe to suggest value changes.
        Only suggest if value is just outside range (possible typo).

        Args:
            finding: The range violation finding
            payload: Original payload

        Returns:
            FixSuggestion if likely typo, None otherwise
        """
        path = finding.path
        actual_value = self._get_value_at_path(payload, path)

        if actual_value is None or not isinstance(actual_value, (int, float)):
            return None

        # Get range constraints
        constraint = self.ir.get_numeric_constraint(path)
        if constraint is None:
            return None

        # Determine the range
        min_val = constraint.minimum
        max_val = constraint.maximum
        excl_min = constraint.exclusive_minimum
        excl_max = constraint.exclusive_maximum

        effective_min = excl_min if excl_min is not None else min_val
        effective_max = excl_max if excl_max is not None else max_val

        if effective_min is None and effective_max is None:
            return None

        # Calculate range span for threshold
        if effective_min is not None and effective_max is not None:
            range_span = abs(effective_max - effective_min)
        elif effective_min is not None:
            range_span = abs(effective_min) if effective_min != 0 else 1.0
        else:
            range_span = abs(effective_max) if effective_max else 1.0

        threshold = range_span * RANGE_TYPO_THRESHOLD

        # Check if value is just outside range (possible typo)
        suggested_value = None
        distance_from_bound = None

        if effective_max is not None and actual_value > effective_max:
            distance_from_bound = actual_value - effective_max
            if distance_from_bound <= threshold:
                # Suggest clamping to max
                suggested_value = effective_max
        elif effective_min is not None and actual_value < effective_min:
            distance_from_bound = effective_min - actual_value
            if distance_from_bound <= threshold:
                # Suggest clamping to min
                suggested_value = effective_min

        if suggested_value is None:
            logger.debug(
                f"Value {actual_value} is too far outside range to suggest fix"
            )
            return None

        patches = self._generator.generate_replace(
            path=path,
            old_value=actual_value,
            new_value=suggested_value
        )

        # Range value changes are always unsafe or needs_review
        safety = PatchSafety.NEEDS_REVIEW
        rationale = (
            f"Value {actual_value} is just outside valid range. "
            f"Suggesting boundary value {suggested_value} (possible typo)"
        )
        confidence = 0.5  # Low confidence for value changes

        context = PatchContext(
            finding=finding,
            original_value=actual_value,
            suggested_value=suggested_value,
            derivation="range_boundary_clamp"
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    def suggest_for_enum_violation(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E202 (enum violation).

        Suggests closest enum value if edit distance <= 2.
        Safety: needs_review.

        Args:
            finding: The enum violation finding
            payload: Original payload

        Returns:
            FixSuggestion if close match found, None otherwise
        """
        path = finding.path
        actual_value = self._get_value_at_path(payload, path)

        if actual_value is None:
            return None

        # Get allowed enum values
        enum_values = self.ir.get_enum(path)
        if not enum_values:
            # Try from finding expected
            expected = finding.expected or {}
            enum_values = expected.get("enum", [])

        if not enum_values:
            return None

        # Convert to string for comparison
        actual_str = str(actual_value)
        enum_str_values = {str(v): v for v in enum_values}

        # Find closest match
        close_matches = self._find_close_matches(
            key=actual_str,
            known_keys=set(enum_str_values.keys()),
            max_distance=MAX_ENUM_DISTANCE
        )

        if not close_matches:
            return None

        best_match_str, distance = close_matches[0]
        suggested_value = enum_str_values[best_match_str]

        patches = self._generator.generate_replace(
            path=path,
            old_value=actual_value,
            new_value=suggested_value
        )

        safety = PatchSafety.NEEDS_REVIEW
        rationale = (
            f"Invalid enum value '{actual_value}'. "
            f"Closest match: '{suggested_value}' (edit distance: {distance})"
        )
        confidence = 0.8 - (distance * 0.15)

        context = PatchContext(
            finding=finding,
            original_value=actual_value,
            suggested_value=suggested_value,
            derivation="enum_closest_match",
            edit_distance=distance
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    # -------------------------------------------------------------------------
    # Unit Error Heuristics (E3xx)
    # -------------------------------------------------------------------------

    def suggest_for_unit_incompatible(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E301 (unit incompatible).

        Cannot fix dimension mismatch (kg vs kWh).
        May suggest if user provided wrong unit string.

        Args:
            finding: The unit incompatible finding
            payload: Original payload

        Returns:
            FixSuggestion if typo in unit, None for true dimension mismatch
        """
        path = finding.path
        actual = finding.actual

        if not actual or not isinstance(actual, dict):
            return None

        actual_unit = actual.get("unit")
        if not actual_unit:
            return None

        # Get unit spec
        unit_spec = self.ir.get_unit_spec(path)
        if unit_spec is None:
            return None

        # Get allowed units for the correct dimension
        allowed_units = set(unit_spec.allowed) if unit_spec.allowed else set()
        if unit_spec.canonical:
            allowed_units.add(unit_spec.canonical)

        if not allowed_units:
            return None

        # Check if actual unit is close to an allowed unit (typo)
        close_matches = self._find_close_matches(
            key=str(actual_unit),
            known_keys=allowed_units,
            max_distance=MAX_TYPO_DISTANCE
        )

        if not close_matches:
            # True dimension mismatch - cannot fix
            logger.debug(
                f"Unit '{actual_unit}' has dimension mismatch, cannot fix"
            )
            return None

        best_match, distance = close_matches[0]

        # This is a unit typo - build path to unit field
        unit_path = f"{path}/unit"
        actual_payload_value = self._get_value_at_path(payload, unit_path)

        if actual_payload_value is None:
            # Unit might be in a different structure
            return None

        patches = self._generator.generate_replace(
            path=unit_path,
            old_value=actual_unit,
            new_value=best_match
        )

        safety = PatchSafety.NEEDS_REVIEW
        rationale = (
            f"Possible unit typo: '{actual_unit}' -> '{best_match}' "
            f"(expected dimension: {unit_spec.dimension})"
        )
        confidence = 0.7 - (distance * 0.1)

        context = PatchContext(
            finding=finding,
            original_value=actual_unit,
            suggested_value=best_match,
            derivation="unit_typo_correction",
            edit_distance=distance
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    def suggest_for_unit_noncanonical(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-E302 (unit noncanonical).

        Suggests conversion to canonical unit.
        Safety: safe if same dimension.

        Args:
            finding: The unit noncanonical finding
            payload: Original payload

        Returns:
            FixSuggestion with unit conversion, None if cannot convert
        """
        path = finding.path
        actual = finding.actual

        if not actual or not isinstance(actual, dict):
            return None

        actual_value = actual.get("value")
        actual_unit = actual.get("unit")

        if actual_value is None or actual_unit is None:
            return None

        # Get unit spec
        unit_spec = self.ir.get_unit_spec(path)
        if unit_spec is None:
            return None

        canonical_unit = unit_spec.canonical
        if not canonical_unit:
            return None

        # Get expected canonical value from finding (if available)
        expected = finding.expected or {}
        canonical_value = expected.get("canonical_value")

        if canonical_value is None:
            # We would need the UnitCatalog to calculate conversion
            # For now, return None if conversion factor not provided
            logger.debug(
                f"Cannot suggest unit conversion without conversion factor"
            )
            return None

        # Build patches for both value and unit
        patches = []

        # Replace value
        value_path = f"{path}/value"
        patches.extend(self._generator.generate_replace(
            path=value_path,
            old_value=actual_value,
            new_value=canonical_value
        ))

        # Replace unit
        unit_path = f"{path}/unit"
        patches.extend(self._generator.generate_replace(
            path=unit_path,
            old_value=actual_unit,
            new_value=canonical_unit
        ))

        # Same-dimension conversion is safe
        safety = PatchSafety.SAFE
        rationale = (
            f"Converting {actual_value} {actual_unit} to "
            f"{canonical_value} {canonical_unit} (same dimension: {unit_spec.dimension})"
        )
        confidence = 0.95

        context = PatchContext(
            finding=finding,
            original_value=actual,
            suggested_value={"value": canonical_value, "unit": canonical_unit},
            derivation="unit_canonicalization",
            is_unit_conversion=True
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    # -------------------------------------------------------------------------
    # Deprecation Warning Heuristics (W6xx)
    # -------------------------------------------------------------------------

    def suggest_for_deprecated_field(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-W600 (deprecated field).

        If replacement field specified, suggest move.
        Safety: safe if replacement declared.

        Args:
            finding: The deprecated field finding
            payload: Original payload

        Returns:
            FixSuggestion if replacement exists, None otherwise
        """
        path = finding.path
        deprecation_info = self.ir.get_deprecation_info(path)

        if deprecation_info is None:
            return None

        replacement = deprecation_info.get("replacement")
        if not replacement:
            logger.debug(f"Deprecated field {path} has no replacement specified")
            return None

        # Check if replacement field already exists
        replacement_value = self._get_value_at_path(payload, replacement)
        if replacement_value is not None:
            logger.debug(
                f"Replacement field '{replacement}' already has a value"
            )
            return None

        patches = self._generator.generate_field_rename(path, replacement)

        safety = PatchSafety.SAFE
        rationale = (
            f"Moving deprecated field '{path}' to replacement '{replacement}'"
        )
        confidence = 0.95

        context = PatchContext(
            finding=finding,
            original_value=None,
            suggested_value=None,
            derivation="deprecation_migration",
            is_alias_resolution=True
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    def suggest_for_renamed_field(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Suggest fix for GLSCHEMA-W601 (renamed field).

        Move from old name to new name.
        Safety: safe (schema explicitly declares rename).

        Args:
            finding: The renamed field finding
            payload: Original payload

        Returns:
            FixSuggestion with rename operation
        """
        path = finding.path
        old_key = self._get_key_from_path(path)

        if not old_key:
            return None

        # Look up the new name
        new_key = self.ir.get_renamed_to(old_key)
        if not new_key:
            # Try from finding expected
            expected = finding.expected or {}
            new_key = expected.get("new_name")

        if not new_key:
            return None

        new_path = self._replace_last_key(path, new_key)

        # Check if new field already exists
        new_value = self._get_value_at_path(payload, new_path)
        if new_value is not None:
            logger.debug(f"New field '{new_path}' already has a value")
            return None

        patches = self._generator.generate_field_rename(path, new_path)

        # Schema-declared rename is safe
        safety = PatchSafety.SAFE
        rationale = (
            f"Renaming field '{old_key}' to '{new_key}' per schema definition"
        )
        confidence = 1.0

        context = PatchContext(
            finding=finding,
            original_value=None,
            suggested_value=None,
            derivation="schema_rename",
            is_alias_resolution=True
        )

        return self._build_suggestion(
            patches=patches,
            context=context,
            finding=finding,
            safety_override=safety,
            confidence=confidence,
            rationale=rationale
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _find_close_matches(
        self,
        key: str,
        known_keys: Set[str],
        max_distance: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Find keys within edit distance.

        Args:
            key: The key to find matches for
            known_keys: Set of known valid keys
            max_distance: Maximum Levenshtein distance

        Returns:
            List of (matched_key, distance) tuples, sorted by distance
        """
        matches = []

        for known_key in known_keys:
            distance = self._levenshtein_distance(key, known_key)
            if distance <= max_distance:
                matches.append((known_key, distance))

        # Sort by distance, then alphabetically
        matches.sort(key=lambda x: (x[1], x[0]))

        return matches

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.

        Uses dynamic programming for O(nm) time complexity.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance (number of insertions, deletions, substitutions)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _can_coerce(
        self,
        value: Any,
        target_type: str
    ) -> Tuple[bool, Any]:
        """
        Check if value can be safely coerced to target type.

        Safe coercions are exact (no data loss):
        - "42" -> 42 for integer
        - "3.14" -> 3.14 for number
        - "true"/"false" -> True/False for boolean

        Args:
            value: The value to coerce
            target_type: The target type string

        Returns:
            Tuple of (can_coerce, coerced_value)
        """
        if not isinstance(value, str):
            return (False, None)

        value_lower = value.lower().strip()

        if target_type == "integer":
            # Try exact integer parsing
            try:
                coerced = int(value)
                # Verify exact conversion
                if str(coerced) == value.strip():
                    return (True, coerced)
            except ValueError:
                pass
            return (False, None)

        elif target_type == "number":
            # Try exact number parsing
            try:
                coerced = float(value)
                # Verify reasonable precision
                if str(coerced) == value.strip() or str(int(coerced)) == value.strip():
                    return (True, coerced)
                # Allow scientific notation
                if 'e' in value.lower() or 'E' in value:
                    return (True, coerced)
            except ValueError:
                pass
            return (False, None)

        elif target_type == "boolean":
            if value_lower in BOOLEAN_TRUE_VALUES:
                return (True, True)
            elif value_lower in BOOLEAN_FALSE_VALUES:
                return (True, False)
            return (False, None)

        elif target_type == "null":
            if value_lower == "null" or value_lower == "none":
                return (True, None)
            return (False, None)

        return (False, None)

    def _build_suggestion(
        self,
        patches: List[JSONPatchOperation],
        context: PatchContext,
        finding: Finding,
        safety_override: Optional[PatchSafety] = None,
        confidence: float = 0.5,
        rationale: str = ""
    ) -> FixSuggestion:
        """
        Build FixSuggestion with safety classification.

        Args:
            patches: List of patch operations
            context: Patch context
            finding: Original finding
            safety_override: Override safety classification (optional)
            confidence: Confidence score
            rationale: Human-readable rationale

        Returns:
            Complete FixSuggestion
        """
        # Separate test operations as preconditions
        preconditions = []
        actual_patches = []

        for patch in patches:
            # Convert JSONPatchOperation to model JSONPatchOp
            model_patch = ModelJSONPatchOp(
                op=patch.op.value if isinstance(patch.op, PatchOp) else patch.op,
                path=patch.path,
                value=patch.value,
                from_=patch.from_
            )
            if patch.op == PatchOp.TEST:
                preconditions.append(model_patch)
            else:
                actual_patches.append(model_patch)

        # Determine safety
        if safety_override is not None:
            safety = safety_override
        else:
            # Use classifier for first non-test patch
            if actual_patches:
                try:
                    # Create a JSONPatchOperation for the classifier (using safety module's type)
                    from .safety import PatchOp as SafetyPatchOp
                    patch_for_classifier = SafetyJSONPatchOperation(
                        op=SafetyPatchOp(actual_patches[0].op),
                        path=actual_patches[0].path,
                        value=actual_patches[0].value,
                        from_=actual_patches[0].from_
                    )
                    classification = self._classifier.classify(
                        patch_for_classifier,
                        context
                    )
                    safety = PatchSafety(classification.safety.value)
                except (NotImplementedError, Exception):
                    # Classifier not yet implemented or failed, default to needs_review
                    safety = PatchSafety.NEEDS_REVIEW
            else:
                safety = PatchSafety.SAFE

        return FixSuggestion(
            patch=actual_patches if actual_patches else [
                ModelJSONPatchOp(
                    op=p.op.value if isinstance(p.op, PatchOp) else p.op,
                    path=p.path,
                    value=p.value,
                    from_=p.from_
                )
                for p in patches
            ],
            preconditions=preconditions,
            confidence=confidence,
            safety=safety,
            rationale=rationale
        )

    def _get_value_at_path(
        self,
        payload: Dict[str, Any],
        path: str
    ) -> Any:
        """
        Get value at JSON Pointer path.

        Args:
            payload: The payload document
            path: JSON Pointer path (e.g., "/foo/bar")

        Returns:
            Value at path, or None if not found
        """
        if not path or path == "":
            return payload

        # Remove leading /
        if path.startswith("/"):
            path = path[1:]

        parts = path.split("/")
        current = payload

        for part in parts:
            if not part:
                continue

            # Handle array indices
            if isinstance(current, list):
                try:
                    index = int(part)
                    if 0 <= index < len(current):
                        current = current[index]
                    else:
                        return None
                except ValueError:
                    return None
            elif isinstance(current, dict):
                # Unescape JSON Pointer special characters
                part = part.replace("~1", "/").replace("~0", "~")
                if part in current:
                    current = current[part]
                else:
                    return None
            else:
                return None

        return current

    def _get_key_from_path(self, path: str) -> Optional[str]:
        """
        Extract the last key from a JSON Pointer path.

        Args:
            path: JSON Pointer path (e.g., "/foo/bar")

        Returns:
            Last key (e.g., "bar"), or None if invalid
        """
        if not path:
            return None

        parts = path.rstrip("/").split("/")
        if len(parts) < 2:
            return None

        key = parts[-1]
        # Unescape JSON Pointer special characters
        key = key.replace("~1", "/").replace("~0", "~")
        return key

    def _replace_last_key(self, path: str, new_key: str) -> str:
        """
        Replace the last key in a JSON Pointer path.

        Args:
            path: Original path (e.g., "/foo/bar")
            new_key: New key to use (e.g., "baz")

        Returns:
            New path (e.g., "/foo/baz")
        """
        parts = path.rstrip("/").split("/")
        if len(parts) < 2:
            return f"/{new_key}"

        # Escape JSON Pointer special characters in new key
        escaped_key = new_key.replace("~", "~0").replace("/", "~1")
        parts[-1] = escaped_key

        return "/".join(parts)

    def _get_known_keys_for_parent(self, path: str) -> Set[str]:
        """
        Get known keys for the parent object of a path.

        Args:
            path: JSON Pointer path (e.g., "/parent/field")

        Returns:
            Set of known field names for the parent object
        """
        # Get parent path
        parts = path.rstrip("/").split("/")
        if len(parts) <= 2:
            parent_path = ""
        else:
            parent_path = "/".join(parts[:-1])

        # Find all properties that are children of parent
        known_keys = set()
        prefix = f"{parent_path}/" if parent_path else "/"

        for prop_path in self.ir.properties.keys():
            if prop_path.startswith(prefix):
                # Extract the direct child key
                remainder = prop_path[len(prefix):]
                if "/" not in remainder:
                    # Direct child
                    key = remainder.replace("~1", "/").replace("~0", "~")
                    known_keys.add(key)

        return known_keys


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "MAX_TYPO_DISTANCE",
    "MAX_ENUM_DISTANCE",
    "RANGE_TYPO_THRESHOLD",
    "BOOLEAN_TRUE_VALUES",
    "BOOLEAN_FALSE_VALUES",
    # Main class
    "FixHeuristics",
]
