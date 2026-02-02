# -*- coding: utf-8 -*-
"""
Fix Suggestion Engine for GL-FOUND-X-002.

This module implements the main fix suggestion engine that orchestrates
the generation, filtering, and validation of fix suggestions for schema
validation findings.

The engine:
    1. Generates fix suggestions for validation findings using heuristics
    2. Classifies suggestions by safety level (safe/needs_review/unsafe)
    3. Filters suggestions based on the configured patch level
    4. Validates suggestions are applicable to the payload
    5. Provides preview of patch application results

Design Principles:
    - Zero-hallucination: Only deterministic, schema-backed suggestions
    - Safety-first: Unsafe patches never auto-applied
    - Audit trail: Complete provenance for all suggestions
    - Performance: Efficient batch processing with early exit on fail_fast

Fixable Error Codes:
    - GLSCHEMA-E100: missing_required (if has default)
    - GLSCHEMA-E101: unknown_field (if typo/rename)
    - GLSCHEMA-E102: type_mismatch (if coercible)
    - GLSCHEMA-E200: range_violation (limited)
    - GLSCHEMA-E202: enum_violation (if close match)
    - GLSCHEMA-E301: unit_incompatible (limited - typo only)
    - GLSCHEMA-E302: unit_noncanonical (unit conversion)
    - GLSCHEMA-W600: deprecated_field (if replacement)
    - GLSCHEMA-W601: renamed_field (move operation)
    - GLSCHEMA-W700: suspicious_key (if close match)

Example:
    >>> from greenlang.schema.suggestions.engine import FixSuggestionEngine
    >>> from greenlang.schema.suggestions.engine import generate_suggestions
    >>> engine = FixSuggestionEngine(ir, options)
    >>> result = engine.generate(findings, payload)
    >>> for fix in result.suggestions:
    ...     if fix.safety == PatchSafety.SAFE:
    ...         print(f"Safe fix: {fix.rationale}")

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 4.4
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..compiler.ir import SchemaIR
from ..models.config import ValidationOptions, PatchLevel
from ..models.finding import Finding
from ..models.patch import FixSuggestion, PatchSafety, JSONPatchOp
from .patches import (
    PatchGenerator,
    JSONPatchOperation,
    PatchOp,
    apply_patch,
    get_value_at_pointer,
    PatchApplicationError,
)
from .safety import PatchSafetyClassifier, PatchContext
from .heuristics import FixHeuristics


logger = logging.getLogger(__name__)


# =============================================================================
# FIXABLE ERROR CODES
# =============================================================================

FIXABLE_CODES: Dict[str, str] = {
    "GLSCHEMA-E100": "missing_required",    # If has default
    "GLSCHEMA-E101": "unknown_field",       # If typo/rename
    "GLSCHEMA-E102": "type_mismatch",       # If coercible
    "GLSCHEMA-E200": "range_violation",     # Limited
    "GLSCHEMA-E202": "enum_violation",      # If close match
    "GLSCHEMA-E301": "unit_incompatible",   # Limited (typo only)
    "GLSCHEMA-E302": "unit_noncanonical",   # Unit conversion
    "GLSCHEMA-W600": "deprecated_field",    # If replacement
    "GLSCHEMA-W601": "renamed_field",       # Move operation
    "GLSCHEMA-W700": "suspicious_key",      # If close match (typo)
}
"""
Error codes that can potentially have fix suggestions generated.

Maps error code to a human-readable fix type description.
Not all findings with these codes will have fixes - that depends
on the specific context (e.g., defaults available, close matches found).
"""


# =============================================================================
# RESULT MODEL
# =============================================================================


class SuggestionEngineResult(BaseModel):
    """
    Result from the suggestion engine.

    Contains all generated suggestions along with statistics about
    the generation process.

    Attributes:
        suggestions: List of generated fix suggestions
        total_generated: Total number of suggestions generated before filtering
        filtered_count: Number of suggestions filtered out by safety level
        errors: List of error messages from failed suggestion generation

    Example:
        >>> result = engine.generate(findings, payload)
        >>> print(f"Generated {result.total_generated} suggestions")
        >>> print(f"Filtered {result.filtered_count} by safety level")
        >>> print(f"Final: {len(result.suggestions)} suggestions")
    """

    suggestions: List[FixSuggestion] = Field(
        default_factory=list,
        description="List of generated fix suggestions"
    )

    total_generated: int = Field(
        default=0,
        ge=0,
        description="Total suggestions generated before filtering"
    )

    filtered_count: int = Field(
        default=0,
        ge=0,
        description="Number filtered by safety level"
    )

    errors: List[str] = Field(
        default_factory=list,
        description="Error messages from failed generation"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    @property
    def has_suggestions(self) -> bool:
        """Check if any suggestions were generated."""
        return len(self.suggestions) > 0

    @property
    def has_safe_suggestions(self) -> bool:
        """Check if any safe suggestions are available."""
        return any(s.safety == PatchSafety.SAFE for s in self.suggestions)

    @property
    def safe_suggestions(self) -> List[FixSuggestion]:
        """Get only safe suggestions."""
        return [s for s in self.suggestions if s.safety == PatchSafety.SAFE]

    def format_summary(self) -> str:
        """Format a summary of the result."""
        safe_count = sum(1 for s in self.suggestions if s.safety == PatchSafety.SAFE)
        review_count = sum(1 for s in self.suggestions if s.safety == PatchSafety.NEEDS_REVIEW)
        unsafe_count = sum(1 for s in self.suggestions if s.safety == PatchSafety.UNSAFE)

        return (
            f"Suggestions: {len(self.suggestions)} total "
            f"(safe={safe_count}, needs_review={review_count}, unsafe={unsafe_count}), "
            f"filtered={self.filtered_count}, errors={len(self.errors)}"
        )


# =============================================================================
# FIX SUGGESTION ENGINE
# =============================================================================


class FixSuggestionEngine:
    """
    Main engine for generating fix suggestions.

    This class orchestrates the generation of fix suggestions for validation
    findings. It uses heuristics to generate suggestions, classifies them
    by safety level, and filters based on the configured patch level.

    Attributes:
        ir: Compiled schema Intermediate Representation
        options: Validation options including patch level settings
        _generator: JSON Patch operation generator
        _classifier: Patch safety classifier
        _heuristics: Fix heuristics engine

    Example:
        >>> engine = FixSuggestionEngine(ir, options)
        >>> result = engine.generate(findings, payload)
        >>> for fix in result.suggestions:
        ...     if fix.safety == PatchSafety.SAFE:
        ...         print(f"Safe fix: {fix.rationale}")
    """

    def __init__(
        self,
        ir: SchemaIR,
        options: Optional[ValidationOptions] = None
    ):
        """
        Initialize FixSuggestionEngine.

        Args:
            ir: Compiled schema Intermediate Representation
            options: Validation options (defaults to standard options)
        """
        self.ir = ir
        self.options = options or ValidationOptions()
        self._generator = PatchGenerator()
        self._classifier = PatchSafetyClassifier(ir)
        self._heuristics = FixHeuristics(ir)

        logger.debug(
            "FixSuggestionEngine initialized for schema %s v%s with patch_level=%s",
            ir.schema_id,
            ir.version,
            self.options.patch_level.value
        )

    def generate(
        self,
        findings: List[Finding],
        payload: Dict[str, Any]
    ) -> SuggestionEngineResult:
        """
        Generate fix suggestions for all findings.

        Processes each finding to generate potential fix suggestions,
        classifies them by safety, and filters based on the configured
        patch level.

        Args:
            findings: List of validation findings
            payload: The original payload being validated

        Returns:
            SuggestionEngineResult with suggestions filtered by safety level

        Example:
            >>> result = engine.generate(report.findings, payload)
            >>> for fix in result.safe_suggestions:
            ...     payload = apply_patches(payload, fix.patch)
        """
        start_time = datetime.now()
        logger.info(
            "Generating suggestions for %d findings with patch_level=%s",
            len(findings),
            self.options.patch_level.value
        )

        # Generate suggestions for each finding
        all_suggestions: List[FixSuggestion] = []
        errors: List[str] = []

        for finding in findings:
            if not self._can_fix(finding):
                continue

            try:
                suggestion = self.generate_for_finding(finding, payload)
                if suggestion is not None:
                    all_suggestions.append(suggestion)
            except Exception as e:
                error_msg = f"Failed to generate suggestion for {finding.code} at {finding.path}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

        total_generated = len(all_suggestions)
        logger.debug("Generated %d total suggestions", total_generated)

        # Deduplicate suggestions
        all_suggestions = self.deduplicate_suggestions(all_suggestions)
        logger.debug("After deduplication: %d suggestions", len(all_suggestions))

        # Validate suggestions are applicable
        all_suggestions = self.validate_suggestions(all_suggestions, payload)
        logger.debug("After validation: %d suggestions", len(all_suggestions))

        # Filter by safety level
        filtered_suggestions = self.filter_by_safety(
            all_suggestions,
            self.options.patch_level
        )
        filtered_count = len(all_suggestions) - len(filtered_suggestions)
        logger.debug(
            "After safety filter (%s): %d suggestions (filtered %d)",
            self.options.patch_level.value,
            len(filtered_suggestions),
            filtered_count
        )

        # Sort suggestions
        sorted_suggestions = self.sort_suggestions(filtered_suggestions)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            "Generated %d suggestions in %.2fms (filtered %d, errors %d)",
            len(sorted_suggestions),
            elapsed_ms,
            filtered_count,
            len(errors)
        )

        return SuggestionEngineResult(
            suggestions=sorted_suggestions,
            total_generated=total_generated,
            filtered_count=filtered_count,
            errors=errors
        )

    def generate_for_finding(
        self,
        finding: Finding,
        payload: Dict[str, Any]
    ) -> Optional[FixSuggestion]:
        """
        Generate suggestion for a single finding.

        Dispatches to the appropriate heuristic based on the finding's
        error code.

        Args:
            finding: The validation finding
            payload: The original payload

        Returns:
            FixSuggestion if a fix can be generated, None otherwise

        Example:
            >>> suggestion = engine.generate_for_finding(finding, payload)
            >>> if suggestion and suggestion.is_safe():
            ...     print(f"Can fix: {suggestion.rationale}")
        """
        if not self._can_fix(finding):
            logger.debug(
                "Finding %s at %s is not fixable",
                finding.code,
                finding.path
            )
            return None

        # Delegate to heuristics
        suggestion = self._heuristics.suggest_for_finding(finding, payload)

        if suggestion is not None:
            logger.debug(
                "Generated suggestion for %s at %s: safety=%s, confidence=%.2f",
                finding.code,
                finding.path,
                suggestion.safety.value,
                suggestion.confidence
            )

        return suggestion

    def filter_by_safety(
        self,
        suggestions: List[FixSuggestion],
        max_level: PatchLevel
    ) -> List[FixSuggestion]:
        """
        Filter suggestions by maximum safety level.

        Args:
            suggestions: List of suggestions to filter
            max_level: Maximum safety level to include

        Returns:
            Filtered list of suggestions

        Behavior:
            - If max_level is SAFE, only return safe suggestions
            - If max_level is NEEDS_REVIEW, return safe + needs_review
            - If max_level is UNSAFE, return all suggestions

        Example:
            >>> safe_only = engine.filter_by_safety(suggestions, PatchLevel.SAFE)
            >>> all_except_unsafe = engine.filter_by_safety(suggestions, PatchLevel.NEEDS_REVIEW)
        """
        if max_level == PatchLevel.UNSAFE:
            # Return all suggestions
            return suggestions

        if max_level == PatchLevel.NEEDS_REVIEW:
            # Return safe and needs_review (exclude unsafe)
            return [
                s for s in suggestions
                if s.safety in (PatchSafety.SAFE, PatchSafety.NEEDS_REVIEW)
            ]

        # max_level == PatchLevel.SAFE - only safe suggestions
        return [s for s in suggestions if s.safety == PatchSafety.SAFE]

    def sort_suggestions(
        self,
        suggestions: List[FixSuggestion]
    ) -> List[FixSuggestion]:
        """
        Sort suggestions by priority.

        Sorting order:
            1. Safety (safe first)
            2. Confidence (high first)
            3. Path (alphabetically for determinism)

        Args:
            suggestions: List of suggestions to sort

        Returns:
            Sorted list of suggestions

        Example:
            >>> sorted_suggestions = engine.sort_suggestions(suggestions)
            >>> # First suggestion is safest and most confident
        """
        def sort_key(s: FixSuggestion) -> Tuple[int, float, str]:
            # Lower safety level number = safer = should come first
            safety_order = s.safety.numeric_level()
            # Higher confidence should come first (negative for descending)
            confidence_order = -s.confidence
            # Path for determinism
            path_order = s.affected_paths()[0] if s.affected_paths() else ""
            return (safety_order, confidence_order, path_order)

        return sorted(suggestions, key=sort_key)

    def deduplicate_suggestions(
        self,
        suggestions: List[FixSuggestion]
    ) -> List[FixSuggestion]:
        """
        Remove duplicate suggestions for same path.

        When multiple suggestions target the same path, keeps the one
        with highest confidence and safest rating.

        Args:
            suggestions: List of suggestions to deduplicate

        Returns:
            Deduplicated list of suggestions

        Example:
            >>> unique = engine.deduplicate_suggestions(suggestions)
            >>> # No two suggestions affect the same path
        """
        if not suggestions:
            return []

        # Group by primary affected path
        path_to_suggestions: Dict[str, List[FixSuggestion]] = {}
        for suggestion in suggestions:
            paths = suggestion.affected_paths()
            primary_path = paths[0] if paths else ""

            if primary_path not in path_to_suggestions:
                path_to_suggestions[primary_path] = []
            path_to_suggestions[primary_path].append(suggestion)

        # Keep best suggestion for each path
        result: List[FixSuggestion] = []
        for path, path_suggestions in path_to_suggestions.items():
            if len(path_suggestions) == 1:
                result.append(path_suggestions[0])
            else:
                # Sort by safety (lower = safer) then confidence (higher = better)
                sorted_suggestions = sorted(
                    path_suggestions,
                    key=lambda s: (s.safety.numeric_level(), -s.confidence)
                )
                result.append(sorted_suggestions[0])
                logger.debug(
                    "Deduplicated %d suggestions for path %s, kept: safety=%s, confidence=%.2f",
                    len(path_suggestions),
                    path,
                    sorted_suggestions[0].safety.value,
                    sorted_suggestions[0].confidence
                )

        return result

    def validate_suggestions(
        self,
        suggestions: List[FixSuggestion],
        payload: Dict[str, Any]
    ) -> List[FixSuggestion]:
        """
        Validate suggestions are applicable.

        Checks that preconditions pass and the patch can be applied
        to the current payload state.

        Args:
            suggestions: List of suggestions to validate
            payload: The original payload

        Returns:
            List of suggestions that pass validation

        Example:
            >>> valid = engine.validate_suggestions(suggestions, payload)
            >>> # All returned suggestions can be applied
        """
        valid_suggestions: List[FixSuggestion] = []

        for suggestion in suggestions:
            try:
                # Check preconditions
                if not self._check_preconditions(suggestion, payload):
                    logger.debug(
                        "Suggestion for %s failed precondition check",
                        suggestion.affected_paths()
                    )
                    continue

                # Try to apply the patch (dry run)
                patch_ops = self._convert_to_patch_operations(suggestion)
                _ = apply_patch(payload, patch_ops)

                valid_suggestions.append(suggestion)

            except PatchApplicationError as e:
                logger.debug(
                    "Suggestion for %s not applicable: %s",
                    suggestion.affected_paths(),
                    str(e)
                )
            except Exception as e:
                logger.warning(
                    "Unexpected error validating suggestion for %s: %s",
                    suggestion.affected_paths(),
                    str(e)
                )

        return valid_suggestions

    def preview_application(
        self,
        suggestion: FixSuggestion,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preview what payload would look like after applying suggestion.

        Creates a copy of the payload and applies the suggestion's patches
        to show the result without modifying the original.

        Args:
            suggestion: The fix suggestion to preview
            payload: The original payload

        Returns:
            New payload dict with suggestion applied

        Raises:
            PatchApplicationError: If the suggestion cannot be applied

        Example:
            >>> preview = engine.preview_application(suggestion, payload)
            >>> print(f"After fix: {preview}")
        """
        patch_ops = self._convert_to_patch_operations(suggestion)

        # Apply preconditions first (they're test operations)
        for precondition in suggestion.preconditions:
            precond_op = self._convert_json_patch_op(precondition)
            patch_ops.insert(0, precond_op)

        return apply_patch(payload, patch_ops)

    def _can_fix(self, finding: Finding) -> bool:
        """
        Check if finding type is fixable.

        Args:
            finding: The validation finding

        Returns:
            True if the error code is in the fixable codes list
        """
        return finding.code in FIXABLE_CODES

    def _check_preconditions(
        self,
        suggestion: FixSuggestion,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Check if suggestion's preconditions are met.

        Args:
            suggestion: The fix suggestion
            payload: The original payload

        Returns:
            True if all preconditions pass
        """
        for precondition in suggestion.preconditions:
            # Preconditions are test operations
            if precondition.op != "test":
                continue

            exists, value = get_value_at_pointer(payload, precondition.path)
            if not exists:
                return False

            # Compare values
            if value != precondition.value:
                return False

        return True

    def _convert_to_patch_operations(
        self,
        suggestion: FixSuggestion
    ) -> List[JSONPatchOperation]:
        """
        Convert FixSuggestion patch ops to JSONPatchOperation objects.

        Args:
            suggestion: The fix suggestion

        Returns:
            List of JSONPatchOperation objects
        """
        operations: List[JSONPatchOperation] = []

        for op in suggestion.patch:
            operations.append(self._convert_json_patch_op(op))

        return operations

    def _convert_json_patch_op(self, op: JSONPatchOp) -> JSONPatchOperation:
        """
        Convert a model JSONPatchOp to a patches.JSONPatchOperation.

        Args:
            op: The model patch operation

        Returns:
            The patches module patch operation
        """
        return JSONPatchOperation(
            op=PatchOp(op.op),
            path=op.path,
            value=op.value,
            from_=op.from_
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_suggestions(
    findings: List[Finding],
    payload: Dict[str, Any],
    ir: SchemaIR,
    options: Optional[ValidationOptions] = None
) -> List[FixSuggestion]:
    """
    Generate fix suggestions for validation findings.

    This is a convenience function that creates a FixSuggestionEngine
    and generates suggestions in one call.

    Args:
        findings: List of validation findings
        payload: The original payload being validated
        ir: Compiled schema IR
        options: Validation options (defaults to standard)

    Returns:
        List of fix suggestions filtered by configured patch level

    Example:
        >>> suggestions = generate_suggestions(report.findings, payload, ir)
        >>> for fix in suggestions:
        ...     if fix.safety == PatchSafety.SAFE:
        ...         payload = apply_patches(payload, fix.patch)

    Note:
        For repeated suggestion generation, create a FixSuggestionEngine
        instance directly for better performance (avoids recreating
        the heuristics and classifier on each call).
    """
    engine = FixSuggestionEngine(ir, options)
    result = engine.generate(findings, payload)
    return result.suggestions


def apply_suggestions(
    payload: Dict[str, Any],
    suggestions: List[FixSuggestion],
    safety_filter: PatchSafety = PatchSafety.SAFE
) -> Tuple[Dict[str, Any], List[FixSuggestion]]:
    """
    Apply fix suggestions to a payload.

    Applies only suggestions at or above the specified safety level.
    Returns the modified payload and list of applied suggestions.

    Args:
        payload: The original payload
        suggestions: List of fix suggestions
        safety_filter: Minimum safety level to apply (default: SAFE only)

    Returns:
        Tuple of (modified_payload, applied_suggestions)

    Example:
        >>> new_payload, applied = apply_suggestions(payload, suggestions)
        >>> print(f"Applied {len(applied)} fixes")
        >>> for fix in applied:
        ...     print(f"  - {fix.rationale}")
    """
    result = copy.deepcopy(payload)
    applied: List[FixSuggestion] = []

    for suggestion in suggestions:
        # Check safety level
        if suggestion.safety.numeric_level() > safety_filter.numeric_level():
            continue

        try:
            # Convert to patch operations
            patch_ops: List[JSONPatchOperation] = []
            for op in suggestion.patch:
                patch_ops.append(JSONPatchOperation(
                    op=PatchOp(op.op),
                    path=op.path,
                    value=op.value,
                    from_=op.from_
                ))

            # Apply the patch
            result = apply_patch(result, patch_ops)
            applied.append(suggestion)

            logger.debug(
                "Applied suggestion: %s (safety=%s)",
                suggestion.rationale[:50],
                suggestion.safety.value
            )

        except PatchApplicationError as e:
            logger.warning(
                "Failed to apply suggestion: %s - %s",
                suggestion.rationale[:50],
                str(e)
            )

    return result, applied


def get_fixable_codes() -> Dict[str, str]:
    """
    Get the set of error codes that can have fixes generated.

    Returns:
        Dictionary mapping error codes to fix type descriptions

    Example:
        >>> codes = get_fixable_codes()
        >>> if finding.code in codes:
        ...     print(f"Fixable: {codes[finding.code]}")
    """
    return FIXABLE_CODES.copy()


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "FIXABLE_CODES",
    # Models
    "SuggestionEngineResult",
    # Main class
    "FixSuggestionEngine",
    # Convenience functions
    "generate_suggestions",
    "apply_suggestions",
    "get_fixable_codes",
]
