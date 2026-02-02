# -*- coding: utf-8 -*-
"""
Patch Safety Classifier for GL-FOUND-X-002.

This module classifies JSON Patch operations by safety level to determine
whether they can be automatically applied or require human review.

Safety Levels:
    - safe: Mechanically correct, high confidence, strong preconditions.
            Can be auto-applied without review.
    - needs_review: Likely correct but context-dependent.
            Should be reviewed by human before applying.
    - unsafe: Speculative, only emit if explicitly requested.
            Should not be auto-applied without explicit confirmation.

Safety Rules:
    - safe: Add optional field with default, coerce exact primitive,
            apply declared alias, unit conversion within same dimension
    - needs_review: Type coercion with potential precision loss, typo
            correction (edit distance), large unit conversion factors
    - unsafe: Infer missing required value, remove data without clear
            schema signal, speculative value replacement

Design Principles:
    - Zero-hallucination: Classification is deterministic based on context
    - Audit trail: Every classification includes rationale and risks
    - Conservative: When in doubt, classify as needs_review or unsafe
    - Provenance: Track how suggestions were derived

Example:
    >>> from greenlang.schema.suggestions.safety import PatchSafetyClassifier
    >>> from greenlang.schema.compiler.ir import SchemaIR
    >>> classifier = PatchSafetyClassifier(ir)
    >>> classification = classifier.classify(patch, context)
    >>> if classification.safety == PatchSafety.SAFE:
    ...     apply_patch(document, [patch])

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 4.2
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from ..compiler.ir import SchemaIR, PropertyIR
from ..models.finding import Finding

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS: Safety Operation Categories
# =============================================================================

# Operations that are safe to auto-apply
SAFE_OPERATIONS: Dict[str, str] = {
    "add_default_optional": "Adding schema default for optional field",
    "add_from_alias": "Adding field from declared alias",
    "replace_exact_coercion": "Type coercion with exact conversion",
    "replace_unit_same_dimension": "Unit conversion within same dimension",
    "move_declared_rename": "Field rename from schema declaration",
    "remove_empty_field": "Removing empty optional field",
}

# Operations that need human review
NEEDS_REVIEW_OPERATIONS: Dict[str, str] = {
    "replace_lossy_coercion": "Type coercion may lose precision",
    "replace_large_unit_factor": "Large unit conversion factor (>1000x)",
    "move_typo_correction": "Possible typo correction",
    "add_inferred_value": "Value inferred from context",
    "remove_deprecated_field": "Removing deprecated field",
}

# Operations that are unsafe (high risk)
UNSAFE_OPERATIONS: Dict[str, str] = {
    "add_required_no_default": "Cannot infer required field value",
    "remove_unknown_field": "Removing data without schema signal",
    "replace_speculative": "Speculative value replacement",
    "remove_data_loss": "Removing field with data (potential data loss)",
    "move_arbitrary_rename": "Arbitrary field rename without schema support",
}

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD: float = 0.9
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.7
LOW_CONFIDENCE_THRESHOLD: float = 0.5

# Unit conversion thresholds
LARGE_UNIT_FACTOR_THRESHOLD: float = 1000.0


# =============================================================================
# ENUMS
# =============================================================================


class PatchSafety(str, Enum):
    """
    Safety classification for patch operations.

    Defines how safe it is to automatically apply a patch:
    - SAFE: Can be auto-applied without review
    - NEEDS_REVIEW: Should be reviewed by human before applying
    - UNSAFE: Should not be auto-applied (high risk)

    Example:
        >>> safety = PatchSafety.SAFE
        >>> print(safety.allows_auto_apply())
        True
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

    @classmethod
    def from_string(cls, value: str) -> "PatchSafety":
        """
        Create PatchSafety from string value.

        Args:
            value: String value ("safe", "needs_review", "unsafe")

        Returns:
            PatchSafety enum value

        Raises:
            ValueError: If value is not a valid safety level
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid safety level '{value}'. "
                f"Must be one of: safe, needs_review, unsafe"
            )


class PatchOp(str, Enum):
    """
    JSON Patch operation types (RFC 6902).

    Represents the different types of operations that can be
    performed in a JSON Patch document.
    """

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"


# =============================================================================
# MODELS
# =============================================================================


class JSONPatchOperation(BaseModel):
    """
    JSON Patch operation (RFC 6902).

    Represents a single operation in a JSON Patch document that can
    be used to modify a JSON document.

    Attributes:
        op: The operation type (add/remove/replace/move/copy/test)
        path: JSON Pointer (RFC 6901) to the target location
        value: The value for add/replace/test operations
        from_: Source path for move/copy operations

    Example:
        >>> patch = JSONPatchOperation(
        ...     op=PatchOp.REPLACE,
        ...     path="/energy/value",
        ...     value=100
        ... )
    """

    op: PatchOp = Field(
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
    }

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path is a valid JSON Pointer."""
        if v == "":
            return v  # Empty string is valid (root)
        if not v.startswith("/"):
            raise ValueError(
                f"Invalid JSON Pointer path '{v}'. Must start with '/' or be empty."
            )
        return v

    def is_additive(self) -> bool:
        """Check if this operation adds data."""
        return self.op in (PatchOp.ADD, PatchOp.COPY)

    def is_destructive(self) -> bool:
        """Check if this operation removes data."""
        return self.op == PatchOp.REMOVE

    def is_modification(self) -> bool:
        """Check if this operation modifies existing data."""
        return self.op in (PatchOp.REPLACE, PatchOp.MOVE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to RFC 6902 compliant dictionary."""
        result: Dict[str, Any] = {"op": self.op.value, "path": self.path}
        if self.op in (PatchOp.ADD, PatchOp.REPLACE, PatchOp.TEST):
            result["value"] = self.value
        elif self.op in (PatchOp.MOVE, PatchOp.COPY):
            result["from"] = self.from_
        return result


class PatchContext(BaseModel):
    """
    Context for safety classification of a patch operation.

    Provides all the information needed to determine the safety
    level of a patch operation, including the finding it addresses,
    original and suggested values, and derivation method.

    Attributes:
        finding: The validation finding this patch addresses
        original_value: The original value before patch (None if adding)
        suggested_value: The value to set after patch
        derivation: How the suggestion was derived
        schema_has_default: Whether schema defines a default value
        is_required_field: Whether the target field is required
        is_alias_resolution: Whether this is from alias resolution
        is_unit_conversion: Whether this involves unit conversion
        is_type_coercion: Whether this involves type coercion
        unit_conversion_factor: Factor for unit conversion (if applicable)
        edit_distance: Edit distance for typo correction (if applicable)

    Example:
        >>> context = PatchContext(
        ...     finding=finding,
        ...     original_value="100",
        ...     suggested_value=100,
        ...     derivation="exact_type_coercion",
        ...     is_type_coercion=True
        ... )
    """

    finding: Finding = Field(
        ...,
        description="The validation finding this patch addresses"
    )
    original_value: Optional[Any] = Field(
        default=None,
        description="Original value before patch"
    )
    suggested_value: Any = Field(
        ...,
        description="Value to set after patch"
    )
    derivation: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="How the suggestion was derived"
    )
    schema_has_default: bool = Field(
        default=False,
        description="Whether schema defines a default value"
    )
    is_required_field: bool = Field(
        default=False,
        description="Whether the target field is required"
    )
    is_alias_resolution: bool = Field(
        default=False,
        description="Whether this is from alias resolution"
    )
    is_unit_conversion: bool = Field(
        default=False,
        description="Whether this involves unit conversion"
    )
    is_type_coercion: bool = Field(
        default=False,
        description="Whether this involves type coercion"
    )
    unit_conversion_factor: Optional[float] = Field(
        default=None,
        description="Factor for unit conversion (if applicable)"
    )
    edit_distance: Optional[int] = Field(
        default=None,
        description="Edit distance for typo correction (if applicable)"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "arbitrary_types_allowed": True,
    }


class SafetyClassification(BaseModel):
    """
    Classification result for a patch operation.

    Contains the safety level, confidence score, rationale,
    identified risks, and whether human review is required.

    Attributes:
        safety: The safety classification (safe/needs_review/unsafe)
        confidence: Confidence score from 0.0 to 1.0
        rationale: Human-readable explanation of the classification
        risks: List of potential risks if patch is applied
        requires_human_review: Whether human review is required

    Example:
        >>> classification = SafetyClassification(
        ...     safety=PatchSafety.SAFE,
        ...     confidence=0.95,
        ...     rationale="Exact type coercion from string to integer",
        ...     risks=[]
        ... )
    """

    safety: PatchSafety = Field(
        ...,
        description="The safety classification"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Human-readable explanation"
    )
    risks: List[str] = Field(
        default_factory=list,
        description="List of potential risks"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is required"
    )
    operation_category: Optional[str] = Field(
        default=None,
        description="Category key from SAFE/NEEDS_REVIEW/UNSAFE_OPERATIONS"
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
    }

    def __post_init__(self) -> None:
        """Set requires_human_review based on safety level."""
        if self.safety != PatchSafety.SAFE:
            object.__setattr__(self, "requires_human_review", True)

    def is_auto_applicable(self, min_confidence: float = 0.9) -> bool:
        """
        Check if patch can be automatically applied.

        Args:
            min_confidence: Minimum required confidence

        Returns:
            True if safe and confidence meets threshold
        """
        return (
            self.safety == PatchSafety.SAFE
            and self.confidence >= min_confidence
            and not self.requires_human_review
        )

    def format_summary(self) -> str:
        """Format a summary of the classification."""
        risk_str = f" Risks: {', '.join(self.risks)}" if self.risks else ""
        return (
            f"[{self.safety.value.upper()}] "
            f"(confidence: {self.confidence:.0%}): "
            f"{self.rationale}{risk_str}"
        )


# =============================================================================
# PATCH SAFETY CLASSIFIER
# =============================================================================


class PatchSafetyClassifier:
    """
    Classifies patch operations by safety level.

    Determines whether a patch can be safely auto-applied, needs
    human review, or is too risky to apply automatically.

    The classifier uses a combination of:
    - Operation type (add/remove/replace/move)
    - Context (schema defaults, required fields, aliases)
    - Value analysis (type coercion, unit conversion)
    - Edit distance for typo detection

    Attributes:
        ir: Compiled schema Intermediate Representation

    Example:
        >>> classifier = PatchSafetyClassifier(ir)
        >>> classification = classifier.classify(patch, context)
        >>> if classification.safety == PatchSafety.SAFE:
        ...     print("Safe to auto-apply")
    """

    def __init__(self, ir: SchemaIR):
        """
        Initialize PatchSafetyClassifier.

        Args:
            ir: Compiled schema Intermediate Representation
        """
        self.ir = ir
        logger.debug(
            "PatchSafetyClassifier initialized for schema %s v%s",
            ir.schema_id,
            ir.version
        )

    def classify(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> SafetyClassification:
        """
        Classify a patch operation's safety.

        Analyzes the patch operation and context to determine
        the appropriate safety level and confidence.

        Args:
            patch: The JSON Patch operation to classify
            context: Context about the patch (finding, values, derivation)

        Returns:
            SafetyClassification with safety level, confidence, and rationale

        Example:
            >>> classification = classifier.classify(patch, context)
            >>> print(classification.safety)
            PatchSafety.SAFE
        """
        start_time = datetime.now()

        logger.debug(
            "Classifying patch: op=%s, path=%s",
            patch.op.value,
            patch.path
        )

        # Route to appropriate classifier based on operation type
        if patch.op == PatchOp.ADD:
            classification = self._classify_add(patch, context)
        elif patch.op == PatchOp.REPLACE:
            classification = self._classify_replace(patch, context)
        elif patch.op == PatchOp.REMOVE:
            classification = self._classify_remove(patch, context)
        elif patch.op == PatchOp.MOVE:
            classification = self._classify_move(patch, context)
        elif patch.op == PatchOp.COPY:
            classification = self._classify_copy(patch, context)
        elif patch.op == PatchOp.TEST:
            # Test operations are always safe (they don't modify data)
            classification = SafetyClassification(
                safety=PatchSafety.SAFE,
                confidence=1.0,
                rationale="Test operations do not modify data",
                risks=[],
                requires_human_review=False,
                operation_category="test_precondition"
            )
        else:
            # Unknown operation type - unsafe
            classification = SafetyClassification(
                safety=PatchSafety.UNSAFE,
                confidence=0.0,
                rationale=f"Unknown operation type: {patch.op}",
                risks=["Unknown operation may have unexpected effects"],
                requires_human_review=True,
                operation_category="unknown_operation"
            )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(
            "Classification complete: safety=%s, confidence=%.2f, time=%.2fms",
            classification.safety.value,
            classification.confidence,
            elapsed_ms
        )

        return classification

    def _classify_add(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> SafetyClassification:
        """
        Classify an add operation.

        Safe if:
        - Adding optional field with schema default
        - Adding field from alias resolution

        Needs review if:
        - Adding value inferred from context

        Unsafe if:
        - Adding required field without default

        Args:
            patch: The add patch operation
            context: Classification context

        Returns:
            SafetyClassification for the add operation
        """
        risks = self._identify_risks(patch, context)

        # Case 1: Adding from alias resolution (safe)
        if context.is_alias_resolution:
            return SafetyClassification(
                safety=PatchSafety.SAFE,
                confidence=HIGH_CONFIDENCE_THRESHOLD,
                rationale=SAFE_OPERATIONS["add_from_alias"],
                risks=risks,
                requires_human_review=False,
                operation_category="add_from_alias"
            )

        # Case 2: Adding optional field with schema default (safe)
        if context.schema_has_default and not context.is_required_field:
            return SafetyClassification(
                safety=PatchSafety.SAFE,
                confidence=HIGH_CONFIDENCE_THRESHOLD,
                rationale=SAFE_OPERATIONS["add_default_optional"],
                risks=risks,
                requires_human_review=False,
                operation_category="add_default_optional"
            )

        # Case 3: Adding required field with schema default (needs review)
        if context.schema_has_default and context.is_required_field:
            return SafetyClassification(
                safety=PatchSafety.NEEDS_REVIEW,
                confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                rationale=(
                    "Adding required field with default value - "
                    "verify default is appropriate for this context"
                ),
                risks=risks + ["Default value may not be appropriate for all contexts"],
                requires_human_review=True,
                operation_category="add_inferred_value"
            )

        # Case 4: Adding inferred value (needs review)
        if "inferred" in context.derivation.lower():
            return SafetyClassification(
                safety=PatchSafety.NEEDS_REVIEW,
                confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                rationale=NEEDS_REVIEW_OPERATIONS["add_inferred_value"],
                risks=risks + ["Inferred value may not match intended value"],
                requires_human_review=True,
                operation_category="add_inferred_value"
            )

        # Case 5: Adding required field without default (unsafe)
        if context.is_required_field and not context.schema_has_default:
            return SafetyClassification(
                safety=PatchSafety.UNSAFE,
                confidence=LOW_CONFIDENCE_THRESHOLD,
                rationale=UNSAFE_OPERATIONS["add_required_no_default"],
                risks=risks + [
                    "Cannot reliably infer value for required field",
                    "May introduce incorrect data"
                ],
                requires_human_review=True,
                operation_category="add_required_no_default"
            )

        # Default: needs review for any other add operation
        confidence = self._calculate_confidence(PatchSafety.NEEDS_REVIEW, context)
        rationale = self._generate_rationale(
            PatchSafety.NEEDS_REVIEW, patch, context
        )

        return SafetyClassification(
            safety=PatchSafety.NEEDS_REVIEW,
            confidence=confidence,
            rationale=rationale,
            risks=risks,
            requires_human_review=True,
            operation_category="add_inferred_value"
        )

    def _classify_replace(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> SafetyClassification:
        """
        Classify a replace operation.

        Safe if:
        - Type coercion that is exact and reversible
        - Unit conversion within same dimension

        Needs review if:
        - Type coercion with potential precision loss
        - Unit conversion with large factor

        Unsafe if:
        - Value inference
        - Speculative replacement

        Args:
            patch: The replace patch operation
            context: Classification context

        Returns:
            SafetyClassification for the replace operation
        """
        risks = self._identify_risks(patch, context)

        # Case 1: Exact type coercion (safe)
        if context.is_type_coercion:
            is_exact = self._is_exact_coercion(
                context.original_value,
                context.suggested_value
            )
            if is_exact:
                return SafetyClassification(
                    safety=PatchSafety.SAFE,
                    confidence=HIGH_CONFIDENCE_THRESHOLD,
                    rationale=SAFE_OPERATIONS["replace_exact_coercion"],
                    risks=risks,
                    requires_human_review=False,
                    operation_category="replace_exact_coercion"
                )
            else:
                # Lossy coercion - needs review
                return SafetyClassification(
                    safety=PatchSafety.NEEDS_REVIEW,
                    confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                    rationale=NEEDS_REVIEW_OPERATIONS["replace_lossy_coercion"],
                    risks=risks + ["Type coercion may lose precision"],
                    requires_human_review=True,
                    operation_category="replace_lossy_coercion"
                )

        # Case 2: Unit conversion
        if context.is_unit_conversion:
            factor = context.unit_conversion_factor or 1.0
            abs_factor = abs(factor) if factor != 0 else 1.0

            # Check for large conversion factor
            if abs_factor > LARGE_UNIT_FACTOR_THRESHOLD or abs_factor < (1 / LARGE_UNIT_FACTOR_THRESHOLD):
                return SafetyClassification(
                    safety=PatchSafety.NEEDS_REVIEW,
                    confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                    rationale=NEEDS_REVIEW_OPERATIONS["replace_large_unit_factor"],
                    risks=risks + [f"Large conversion factor ({factor}x) may indicate error"],
                    requires_human_review=True,
                    operation_category="replace_large_unit_factor"
                )
            else:
                return SafetyClassification(
                    safety=PatchSafety.SAFE,
                    confidence=HIGH_CONFIDENCE_THRESHOLD,
                    rationale=SAFE_OPERATIONS["replace_unit_same_dimension"],
                    risks=risks,
                    requires_human_review=False,
                    operation_category="replace_unit_same_dimension"
                )

        # Case 3: Speculative replacement (unsafe)
        if "speculative" in context.derivation.lower():
            return SafetyClassification(
                safety=PatchSafety.UNSAFE,
                confidence=LOW_CONFIDENCE_THRESHOLD,
                rationale=UNSAFE_OPERATIONS["replace_speculative"],
                risks=risks + [
                    "Speculative replacement may be incorrect",
                    "Original value will be lost"
                ],
                requires_human_review=True,
                operation_category="replace_speculative"
            )

        # Default: needs review for other replace operations
        confidence = self._calculate_confidence(PatchSafety.NEEDS_REVIEW, context)
        rationale = self._generate_rationale(
            PatchSafety.NEEDS_REVIEW, patch, context
        )

        return SafetyClassification(
            safety=PatchSafety.NEEDS_REVIEW,
            confidence=confidence,
            rationale=rationale,
            risks=risks,
            requires_human_review=True,
            operation_category="replace_speculative"
        )

    def _classify_remove(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> SafetyClassification:
        """
        Classify a remove operation.

        Safe if:
        - Removing truly unknown field in strict mode
        - Removing empty optional field

        Needs review if:
        - Removing field that might be intentional
        - Removing deprecated field

        Unsafe if:
        - Removing any data without clear schema signal

        Args:
            patch: The remove patch operation
            context: Classification context

        Returns:
            SafetyClassification for the remove operation
        """
        risks = self._identify_risks(patch, context)

        # Get the field name from path
        field_name = patch.path.split("/")[-1] if "/" in patch.path else patch.path

        # Case 1: Check if field is in schema
        prop = self.ir.get_property(patch.path)
        is_known_field = prop is not None

        # Case 2: Check if removing empty/null value
        is_empty = (
            context.original_value is None
            or context.original_value == ""
            or context.original_value == []
            or context.original_value == {}
        )

        # Case 3: Check if deprecated
        is_deprecated = self.ir.is_deprecated(patch.path)

        # Safe: Removing empty optional field
        if is_empty and not context.is_required_field:
            return SafetyClassification(
                safety=PatchSafety.SAFE,
                confidence=HIGH_CONFIDENCE_THRESHOLD,
                rationale=SAFE_OPERATIONS["remove_empty_field"],
                risks=risks,
                requires_human_review=False,
                operation_category="remove_empty_field"
            )

        # Needs review: Removing deprecated field
        if is_deprecated:
            return SafetyClassification(
                safety=PatchSafety.NEEDS_REVIEW,
                confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                rationale=NEEDS_REVIEW_OPERATIONS["remove_deprecated_field"],
                risks=risks + [
                    "Deprecated field may still be used by downstream systems"
                ],
                requires_human_review=True,
                operation_category="remove_deprecated_field"
            )

        # Safe: Unknown field in strict mode (linting suggestion)
        if not is_known_field and "unknown" in context.derivation.lower():
            # Still needs review if there's data
            if not is_empty:
                return SafetyClassification(
                    safety=PatchSafety.UNSAFE,
                    confidence=LOW_CONFIDENCE_THRESHOLD,
                    rationale=UNSAFE_OPERATIONS["remove_data_loss"],
                    risks=risks + [
                        f"Removing '{field_name}' will lose data: {context.original_value}",
                        "Unknown fields may contain intentional data"
                    ],
                    requires_human_review=True,
                    operation_category="remove_data_loss"
                )
            else:
                return SafetyClassification(
                    safety=PatchSafety.NEEDS_REVIEW,
                    confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                    rationale=UNSAFE_OPERATIONS["remove_unknown_field"],
                    risks=risks + ["Unknown field may be intentional"],
                    requires_human_review=True,
                    operation_category="remove_unknown_field"
                )

        # Unsafe: Removing any data without clear signal
        return SafetyClassification(
            safety=PatchSafety.UNSAFE,
            confidence=LOW_CONFIDENCE_THRESHOLD,
            rationale=UNSAFE_OPERATIONS["remove_data_loss"],
            risks=risks + [
                "Removing data may cause data loss",
                "Verify removal is intentional"
            ],
            requires_human_review=True,
            operation_category="remove_data_loss"
        )

    def _classify_move(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> SafetyClassification:
        """
        Classify a move operation (rename).

        Safe if:
        - Schema declares explicit alias/rename mapping

        Needs review if:
        - Typo correction based on edit distance

        Unsafe if:
        - Arbitrary rename without schema support

        Args:
            patch: The move patch operation
            context: Classification context

        Returns:
            SafetyClassification for the move operation
        """
        risks = self._identify_risks(patch, context)

        if patch.from_ is None:
            return SafetyClassification(
                safety=PatchSafety.UNSAFE,
                confidence=0.0,
                rationale="Move operation missing 'from' field",
                risks=risks + ["Invalid move operation"],
                requires_human_review=True,
                operation_category="move_arbitrary_rename"
            )

        # Case 1: Schema-defined rename (safe)
        is_schema_rename = self._is_schema_defined_rename(
            patch.from_,
            patch.path,
            context
        )
        if is_schema_rename:
            return SafetyClassification(
                safety=PatchSafety.SAFE,
                confidence=HIGH_CONFIDENCE_THRESHOLD,
                rationale=SAFE_OPERATIONS["move_declared_rename"],
                risks=risks,
                requires_human_review=False,
                operation_category="move_declared_rename"
            )

        # Case 2: Typo correction (needs review)
        if context.edit_distance is not None and context.edit_distance <= 2:
            old_key = patch.from_.split("/")[-1]
            new_key = patch.path.split("/")[-1]
            return SafetyClassification(
                safety=PatchSafety.NEEDS_REVIEW,
                confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                rationale=(
                    f"{NEEDS_REVIEW_OPERATIONS['move_typo_correction']}: "
                    f"'{old_key}' -> '{new_key}' (edit distance: {context.edit_distance})"
                ),
                risks=risks + [
                    f"Typo correction may be incorrect",
                    f"Original key '{old_key}' may be intentional"
                ],
                requires_human_review=True,
                operation_category="move_typo_correction"
            )

        # Case 3: Alias resolution (safe)
        if context.is_alias_resolution:
            return SafetyClassification(
                safety=PatchSafety.SAFE,
                confidence=HIGH_CONFIDENCE_THRESHOLD,
                rationale=SAFE_OPERATIONS["add_from_alias"],
                risks=risks,
                requires_human_review=False,
                operation_category="move_declared_rename"
            )

        # Default: Arbitrary rename (unsafe)
        return SafetyClassification(
            safety=PatchSafety.UNSAFE,
            confidence=LOW_CONFIDENCE_THRESHOLD,
            rationale=UNSAFE_OPERATIONS["move_arbitrary_rename"],
            risks=risks + [
                "Arbitrary rename may break downstream systems",
                "No schema support for this rename"
            ],
            requires_human_review=True,
            operation_category="move_arbitrary_rename"
        )

    def _classify_copy(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> SafetyClassification:
        """
        Classify a copy operation.

        Copy operations are generally safe as they don't destroy data,
        but may need review for certain scenarios.

        Args:
            patch: The copy patch operation
            context: Classification context

        Returns:
            SafetyClassification for the copy operation
        """
        risks = self._identify_risks(patch, context)

        if patch.from_ is None:
            return SafetyClassification(
                safety=PatchSafety.UNSAFE,
                confidence=0.0,
                rationale="Copy operation missing 'from' field",
                risks=risks + ["Invalid copy operation"],
                requires_human_review=True,
                operation_category="unknown_operation"
            )

        # Copy operations are generally safe as they don't destroy data
        # but we should still review if copying to a location that exists
        target_prop = self.ir.get_property(patch.path)

        if target_prop is not None:
            return SafetyClassification(
                safety=PatchSafety.NEEDS_REVIEW,
                confidence=MEDIUM_CONFIDENCE_THRESHOLD,
                rationale="Copying to existing field location may overwrite data",
                risks=risks + ["Existing value at target location may be overwritten"],
                requires_human_review=True,
                operation_category="add_inferred_value"
            )

        return SafetyClassification(
            safety=PatchSafety.SAFE,
            confidence=HIGH_CONFIDENCE_THRESHOLD,
            rationale="Copy operation does not destroy data",
            risks=risks,
            requires_human_review=False,
            operation_category="add_from_alias"
        )

    def _calculate_confidence(
        self,
        safety: PatchSafety,
        context: PatchContext
    ) -> float:
        """
        Calculate confidence score based on context.

        Considers factors like:
        - Whether schema has a default
        - Whether it's an alias resolution
        - Whether it's a known derivation type
        - Edit distance for typo corrections

        Args:
            safety: The determined safety level
            context: Classification context

        Returns:
            Confidence score from 0.0 to 1.0
        """
        base_confidence = {
            PatchSafety.SAFE: HIGH_CONFIDENCE_THRESHOLD,
            PatchSafety.NEEDS_REVIEW: MEDIUM_CONFIDENCE_THRESHOLD,
            PatchSafety.UNSAFE: LOW_CONFIDENCE_THRESHOLD,
        }.get(safety, 0.5)

        # Boost confidence for schema-backed operations
        if context.schema_has_default:
            base_confidence = min(1.0, base_confidence + 0.1)

        if context.is_alias_resolution:
            base_confidence = min(1.0, base_confidence + 0.1)

        # Reduce confidence for inferred values
        if "inferred" in context.derivation.lower():
            base_confidence = max(0.1, base_confidence - 0.2)

        # Reduce confidence based on edit distance
        if context.edit_distance is not None:
            if context.edit_distance > 2:
                base_confidence = max(0.1, base_confidence - 0.3)
            elif context.edit_distance > 1:
                base_confidence = max(0.1, base_confidence - 0.1)

        return round(base_confidence, 2)

    def _generate_rationale(
        self,
        safety: PatchSafety,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> str:
        """
        Generate human-readable rationale for the classification.

        Args:
            safety: The determined safety level
            patch: The patch operation
            context: Classification context

        Returns:
            Human-readable rationale string
        """
        parts = []

        # Describe the operation
        op_desc = {
            PatchOp.ADD: "Adding",
            PatchOp.REMOVE: "Removing",
            PatchOp.REPLACE: "Replacing",
            PatchOp.MOVE: "Moving/renaming",
            PatchOp.COPY: "Copying",
            PatchOp.TEST: "Testing",
        }.get(patch.op, "Modifying")

        parts.append(f"{op_desc} field at '{patch.path}'")

        # Add derivation info
        if context.derivation:
            parts.append(f"derived from: {context.derivation}")

        # Add safety-specific info
        if safety == PatchSafety.SAFE:
            if context.schema_has_default:
                parts.append("using schema default value")
            if context.is_alias_resolution:
                parts.append("resolving schema-defined alias")
            if context.is_type_coercion:
                parts.append("with exact type coercion")
        elif safety == PatchSafety.NEEDS_REVIEW:
            parts.append("requires human verification")
        elif safety == PatchSafety.UNSAFE:
            parts.append("high risk - manual review required")

        return ". ".join(parts)

    def _identify_risks(
        self,
        patch: JSONPatchOperation,
        context: PatchContext
    ) -> List[str]:
        """
        Identify potential risks of applying the patch.

        Args:
            patch: The patch operation
            context: Classification context

        Returns:
            List of identified risk strings
        """
        risks: List[str] = []

        # Risk: Data loss for remove/replace operations
        if patch.op in (PatchOp.REMOVE, PatchOp.REPLACE):
            if context.original_value is not None:
                risks.append("Original value will be modified or lost")

        # Risk: Required field manipulation
        if context.is_required_field:
            risks.append("Modifying required field may affect validation")

        # Risk: Type coercion
        if context.is_type_coercion:
            if not self._is_exact_coercion(
                context.original_value,
                context.suggested_value
            ):
                risks.append("Type coercion may lose information")

        # Risk: Large unit conversion
        if context.is_unit_conversion and context.unit_conversion_factor:
            factor = abs(context.unit_conversion_factor)
            if factor > LARGE_UNIT_FACTOR_THRESHOLD:
                risks.append(
                    f"Large conversion factor ({factor}x) "
                    "may indicate unit mismatch"
                )

        # Risk: High edit distance for typo corrections
        if context.edit_distance is not None and context.edit_distance > 1:
            risks.append(
                f"Edit distance of {context.edit_distance} may indicate "
                "incorrect typo correction"
            )

        return risks

    def _is_exact_coercion(
        self,
        original: Any,
        suggested: Any
    ) -> bool:
        """
        Check if coercion is exact (no data loss).

        Examples of exact coercions:
        - "42" -> 42
        - "3.14" -> 3.14
        - "true" -> True
        - "false" -> False
        - 42 -> "42" (but this is less common direction)

        Args:
            original: Original value
            suggested: Suggested value after coercion

        Returns:
            True if coercion is exact and reversible
        """
        if original is None or suggested is None:
            return False

        # String to boolean coercion (check before int because bool is subclass of int)
        if isinstance(original, str) and isinstance(suggested, bool):
            lower = original.lower().strip()
            if lower == "true" and suggested is True:
                return True
            if lower == "false" and suggested is False:
                return True
            return False

        # Boolean to string coercion (reverse)
        if isinstance(original, bool) and isinstance(suggested, str):
            lower = suggested.lower().strip()
            if original is True and lower == "true":
                return True
            if original is False and lower == "false":
                return True
            return False

        # String to number coercion (after bool check since bool is subclass of int)
        if isinstance(original, str) and isinstance(suggested, (int, float)) and not isinstance(suggested, bool):
            try:
                # Check if converting back gives the same string
                if isinstance(suggested, int):
                    return str(suggested) == original.strip()
                elif isinstance(suggested, float):
                    # Handle various float representations
                    original_stripped = original.strip()
                    # Check exact match
                    if str(suggested) == original_stripped:
                        return True
                    # Check with scientific notation
                    if f"{suggested:.15g}" == original_stripped:
                        return True
                    # Check if parsed value equals suggested
                    try:
                        parsed = float(original_stripped)
                        return parsed == suggested
                    except ValueError:
                        return False
            except (ValueError, TypeError):
                return False

        # Number to string coercion (less common but should be exact)
        if isinstance(original, (int, float)) and not isinstance(original, bool) and isinstance(suggested, str):
            try:
                parsed = type(original)(suggested.strip())
                return parsed == original
            except (ValueError, TypeError):
                return False

        # Same type, same value
        if type(original) == type(suggested):
            if isinstance(original, float):
                # Use approximate comparison for floats
                return math.isclose(original, suggested, rel_tol=1e-9)
            return original == suggested

        return False

    def _is_schema_defined_rename(
        self,
        old_path: str,
        new_path: str,
        context: PatchContext
    ) -> bool:
        """
        Check if rename is defined in schema (renamed_from).

        Args:
            old_path: Old field path
            new_path: New field path
            context: Classification context

        Returns:
            True if schema defines this rename
        """
        # Extract field names from paths
        old_key = old_path.split("/")[-1] if "/" in old_path else old_path
        new_key = new_path.split("/")[-1] if "/" in new_path else new_path

        # Check renamed_fields map
        renamed_to = self.ir.renamed_fields.get(old_key)
        if renamed_to == new_key:
            return True

        # Check the reverse (new_key might have renamed_from metadata)
        for old_name, new_name in self.ir.renamed_fields.items():
            if old_name == old_key and new_name == new_key:
                return True

        return False


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def is_safe_patch(
    patch: JSONPatchOperation,
    context: PatchContext,
    ir: SchemaIR
) -> bool:
    """
    Quick check if a patch is safe to auto-apply.

    This is a convenience function that creates a classifier
    and checks if the patch is safe.

    Args:
        patch: The JSON Patch operation to check
        context: Context about the patch
        ir: Compiled schema IR

    Returns:
        True if the patch is classified as SAFE

    Example:
        >>> if is_safe_patch(patch, context, ir):
        ...     apply_patch(document, [patch])
    """
    classifier = PatchSafetyClassifier(ir)
    classification = classifier.classify(patch, context)
    return classification.safety == PatchSafety.SAFE


def classify_patches(
    patches: List[JSONPatchOperation],
    contexts: List[PatchContext],
    ir: SchemaIR
) -> List[SafetyClassification]:
    """
    Classify multiple patches.

    Args:
        patches: List of JSON Patch operations
        contexts: List of corresponding contexts
        ir: Compiled schema IR

    Returns:
        List of SafetyClassification results

    Raises:
        ValueError: If patches and contexts lists have different lengths
    """
    if len(patches) != len(contexts):
        raise ValueError(
            f"patches ({len(patches)}) and contexts ({len(contexts)}) "
            "must have the same length"
        )

    classifier = PatchSafetyClassifier(ir)
    return [
        classifier.classify(patch, context)
        for patch, context in zip(patches, contexts)
    ]


def filter_safe_patches(
    patches: List[JSONPatchOperation],
    contexts: List[PatchContext],
    ir: SchemaIR,
    min_confidence: float = HIGH_CONFIDENCE_THRESHOLD
) -> List[Tuple[JSONPatchOperation, SafetyClassification]]:
    """
    Filter patches to return only safe ones.

    Args:
        patches: List of JSON Patch operations
        contexts: List of corresponding contexts
        ir: Compiled schema IR
        min_confidence: Minimum confidence threshold

    Returns:
        List of (patch, classification) tuples for safe patches
    """
    if len(patches) != len(contexts):
        raise ValueError(
            f"patches ({len(patches)}) and contexts ({len(contexts)}) "
            "must have the same length"
        )

    classifier = PatchSafetyClassifier(ir)
    results = []

    for patch, context in zip(patches, contexts):
        classification = classifier.classify(patch, context)
        if classification.is_auto_applicable(min_confidence):
            results.append((patch, classification))

    return results


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Constants
    "SAFE_OPERATIONS",
    "NEEDS_REVIEW_OPERATIONS",
    "UNSAFE_OPERATIONS",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MEDIUM_CONFIDENCE_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    "LARGE_UNIT_FACTOR_THRESHOLD",
    # Enums
    "PatchSafety",
    "PatchOp",
    # Models
    "JSONPatchOperation",
    "PatchContext",
    "SafetyClassification",
    # Main class
    "PatchSafetyClassifier",
    # Convenience functions
    "is_safe_patch",
    "classify_patches",
    "filter_safe_patches",
]
