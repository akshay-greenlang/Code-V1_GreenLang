"""
Vocabulary validation utilities for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides comprehensive validation for vocabularies including:
- Signature verification for integrity
- Schema validation for structure
- Deprecation checking for outdated entities

Key Design Principles:
    - Zero-hallucination: Deterministic validation rules only
    - Complete audit trail: All validation results are logged
    - Governance support: Deprecation warnings enable proactive updates

Example:
    >>> from gl_normalizer_core.vocabulary.validators import (
    ...     validate_signature,
    ...     validate_schema,
    ...     check_deprecations,
    ... )
    >>> vocab = registry.get_vocabulary("fuels")
    >>> is_valid = validate_signature(vocab, vocab.metadata.signature)
    >>> errors = validate_schema(vocab)
    >>> warnings = check_deprecations(vocab)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import hashlib
import json
import logging
import re

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.vocabulary.models import (
    Vocabulary,
    Entity,
    Alias,
    EntityType,
    VocabularyMetadata,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """
    Structured validation error.

    Attributes:
        code: Error code from GLNORMErrorCode.
        message: Human-readable error message.
        path: JSON path to the affected field.
        severity: Severity level (error, warning, info).
        details: Additional context about the error.
        suggestion: Optional suggestion for resolution.
    """

    code: str
    message: str
    path: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    details: Dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity.value,
            "details": self.details,
            "suggestion": self.suggestion,
        }


@dataclass
class DeprecationWarning:
    """
    Warning about deprecated entities.

    Attributes:
        entity_id: ID of the deprecated entity.
        entity_name: Canonical name of the entity.
        deprecated_at: When the entity was deprecated.
        reason: Reason for deprecation.
        replacement_id: ID of replacement entity, if available.
        removal_date: Planned removal date, if known.
        days_until_removal: Days until removal (if removal_date set).
    """

    entity_id: str
    entity_name: str
    deprecated_at: datetime
    reason: str
    replacement_id: Optional[str] = None
    removal_date: Optional[datetime] = None
    days_until_removal: Optional[int] = None

    def __post_init__(self) -> None:
        """Calculate days until removal if applicable."""
        if self.removal_date:
            delta = self.removal_date - datetime.utcnow()
            self.days_until_removal = max(0, delta.days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "deprecated_at": self.deprecated_at.isoformat(),
            "reason": self.reason,
            "replacement_id": self.replacement_id,
            "removal_date": self.removal_date.isoformat() if self.removal_date else None,
            "days_until_removal": self.days_until_removal,
        }


@dataclass
class ValidationResult:
    """
    Complete validation result for a vocabulary.

    Attributes:
        is_valid: Whether the vocabulary passed validation.
        errors: List of validation errors.
        warnings: List of validation warnings.
        info: List of informational messages.
        validated_at: Timestamp of validation.
        vocabulary_id: ID of the validated vocabulary.
        vocabulary_version: Version of the validated vocabulary.
    """

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    vocabulary_id: Optional[str] = None
    vocabulary_version: Optional[str] = None

    @property
    def error_count(self) -> int:
        """Return the number of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Return the number of warnings."""
        return len(self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "validated_at": self.validated_at.isoformat(),
            "vocabulary_id": self.vocabulary_id,
            "vocabulary_version": self.vocabulary_version,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


def validate_signature(
    vocabulary: Vocabulary,
    expected_signature: Optional[str] = None,
) -> bool:
    """
    Validate a vocabulary's signature matches its content.

    Args:
        vocabulary: The vocabulary to validate.
        expected_signature: Optional expected signature.
                          Uses vocabulary.metadata.signature if not provided.

    Returns:
        True if signature is valid, False otherwise.

    Example:
        >>> vocab = registry.get_vocabulary("fuels")
        >>> is_valid = validate_signature(vocab)
        >>> if not is_valid:
        ...     logger.warning("Vocabulary signature mismatch!")
    """
    signature = expected_signature
    if not signature and vocabulary.metadata:
        signature = vocabulary.metadata.signature

    if not signature:
        logger.warning(
            "No signature available for validation",
            vocab_id=vocabulary.id,
        )
        return False

    computed = vocabulary.compute_signature()
    is_valid = computed == signature

    if not is_valid:
        logger.error(
            "Signature mismatch",
            vocab_id=vocabulary.id,
            expected=signature[:32] + "...",
            computed=computed[:32] + "...",
        )
    else:
        logger.debug(
            "Signature validated",
            vocab_id=vocabulary.id,
        )

    return is_valid


def validate_schema(
    vocabulary: Vocabulary,
    strict: bool = False,
) -> List[ValidationError]:
    """
    Validate vocabulary structure and content.

    Performs comprehensive schema validation including:
    - Required fields presence
    - Field type validation
    - Reference integrity (aliases point to valid entities)
    - Entity ID format validation
    - Duplicate detection

    Args:
        vocabulary: The vocabulary to validate.
        strict: If True, treats warnings as errors.

    Returns:
        List of ValidationError objects.

    Example:
        >>> errors = validate_schema(vocabulary)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"{error.code}: {error.message}")
    """
    errors: List[ValidationError] = []

    # Validate vocabulary-level fields
    errors.extend(_validate_vocabulary_fields(vocabulary))

    # Validate metadata
    if vocabulary.metadata:
        errors.extend(_validate_metadata(vocabulary.metadata, vocabulary.id))

    # Validate entities
    entity_ids: Set[str] = set()
    for entity_id, entity in vocabulary.entities.items():
        # Check for duplicate IDs (shouldn't happen with dict, but verify key matches)
        if entity.id != entity_id:
            errors.append(ValidationError(
                code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                message=f"Entity key '{entity_id}' does not match entity.id '{entity.id}'",
                path=f"/entities/{entity_id}",
                severity=ValidationSeverity.ERROR,
            ))

        if entity.id in entity_ids:
            errors.append(ValidationError(
                code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                message=f"Duplicate entity ID: {entity.id}",
                path=f"/entities/{entity_id}",
                severity=ValidationSeverity.ERROR,
            ))
        entity_ids.add(entity.id)

        # Validate entity
        errors.extend(_validate_entity(entity, vocabulary.id))

    # Validate aliases reference valid entities
    for i, alias in enumerate(vocabulary.aliases):
        if alias.canonical_id not in entity_ids:
            errors.append(ValidationError(
                code=GLNORMErrorCode.E400_REFERENCE_NOT_FOUND.value,
                message=f"Alias '{alias.alias}' references non-existent entity '{alias.canonical_id}'",
                path=f"/aliases/{i}",
                severity=ValidationSeverity.ERROR,
                details={"alias": alias.alias, "canonical_id": alias.canonical_id},
            ))

    # Validate deprecation references
    for entity_id, entity in vocabulary.entities.items():
        if entity.deprecated and entity.deprecation_info:
            replacement_id = entity.deprecation_info.replacement_id
            if replacement_id and replacement_id not in entity_ids:
                errors.append(ValidationError(
                    code=GLNORMErrorCode.E400_REFERENCE_NOT_FOUND.value,
                    message=f"Deprecated entity '{entity_id}' references non-existent replacement '{replacement_id}'",
                    path=f"/entities/{entity_id}/deprecation_info/replacement_id",
                    severity=ValidationSeverity.WARNING if not strict else ValidationSeverity.ERROR,
                    suggestion=f"Ensure replacement entity '{replacement_id}' exists in the vocabulary",
                ))

    # Check for alias collisions
    alias_map: Dict[str, List[str]] = {}
    for entity in vocabulary.entities.values():
        for alias in entity.aliases:
            alias_lower = alias.lower()
            if alias_lower not in alias_map:
                alias_map[alias_lower] = []
            alias_map[alias_lower].append(entity.id)

    for alias in vocabulary.aliases:
        alias_lower = alias.alias.lower()
        if alias_lower not in alias_map:
            alias_map[alias_lower] = []
        alias_map[alias_lower].append(alias.canonical_id)

    for alias_str, entity_ids_list in alias_map.items():
        unique_ids = set(entity_ids_list)
        if len(unique_ids) > 1:
            errors.append(ValidationError(
                code=GLNORMErrorCode.E406_ALIAS_COLLISION.value,
                message=f"Alias '{alias_str}' maps to multiple entities: {sorted(unique_ids)}",
                path="/aliases",
                severity=ValidationSeverity.WARNING,
                details={"alias": alias_str, "entity_ids": sorted(unique_ids)},
                suggestion="Resolve alias collision by assigning different priorities or removing duplicates",
            ))

    logger.info(
        "Schema validation complete",
        vocab_id=vocabulary.id,
        error_count=sum(1 for e in errors if e.severity == ValidationSeverity.ERROR),
        warning_count=sum(1 for e in errors if e.severity == ValidationSeverity.WARNING),
    )

    return errors


def _validate_vocabulary_fields(vocabulary: Vocabulary) -> List[ValidationError]:
    """Validate vocabulary-level required fields."""
    errors: List[ValidationError] = []

    # Validate ID format
    if not re.match(r"^[a-z][a-z0-9_-]*$", vocabulary.id):
        errors.append(ValidationError(
            code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
            message=f"Invalid vocabulary ID format: '{vocabulary.id}'",
            path="/id",
            severity=ValidationSeverity.ERROR,
            suggestion="Vocabulary ID must start with lowercase letter and contain only lowercase letters, numbers, hyphens, and underscores",
        ))

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$", vocabulary.version):
        errors.append(ValidationError(
            code=GLNORMErrorCode.E500_VOCABULARY_VERSION_MISMATCH.value,
            message=f"Invalid version format: '{vocabulary.version}'",
            path="/version",
            severity=ValidationSeverity.WARNING,
            suggestion="Version should follow semantic versioning (e.g., '1.0.0', '2026.01.0')",
        ))

    # Check for empty vocabulary
    if not vocabulary.entities:
        errors.append(ValidationError(
            code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
            message="Vocabulary contains no entities",
            path="/entities",
            severity=ValidationSeverity.WARNING,
            suggestion="Add entities to the vocabulary",
        ))

    return errors


def _validate_metadata(
    metadata: VocabularyMetadata,
    vocab_id: str,
) -> List[ValidationError]:
    """Validate vocabulary metadata."""
    errors: List[ValidationError] = []

    # Check for expired vocabulary
    if metadata.is_expired():
        errors.append(ValidationError(
            code=GLNORMErrorCode.E503_VOCABULARY_EXPIRED.value,
            message=f"Vocabulary expired at {metadata.expires_at}",
            path="/metadata/expires_at",
            severity=ValidationSeverity.WARNING,
            details={"expires_at": metadata.expires_at.isoformat()},
            suggestion="Update to a newer version of the vocabulary",
        ))

    # Validate signature format if present
    if metadata.signature:
        if not re.match(r"^sha256:[a-f0-9]{64}$", metadata.signature):
            errors.append(ValidationError(
                code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                message="Invalid signature format",
                path="/metadata/signature",
                severity=ValidationSeverity.ERROR,
                suggestion="Signature must be in format 'sha256:<64-char-hex>'",
            ))

    return errors


def _validate_entity(entity: Entity, vocab_id: str) -> List[ValidationError]:
    """Validate a single entity."""
    errors: List[ValidationError] = []
    entity_path = f"/entities/{entity.id}"

    # Validate ID format
    if not re.match(r"^[A-Za-z0-9_-]+$", entity.id):
        errors.append(ValidationError(
            code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
            message=f"Invalid entity ID format: '{entity.id}'",
            path=f"{entity_path}/id",
            severity=ValidationSeverity.ERROR,
            suggestion="Entity ID must contain only letters, numbers, hyphens, and underscores",
        ))

    # Validate canonical name is not empty
    if not entity.canonical_name or not entity.canonical_name.strip():
        errors.append(ValidationError(
            code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
            message=f"Entity '{entity.id}' has empty canonical name",
            path=f"{entity_path}/canonical_name",
            severity=ValidationSeverity.ERROR,
        ))

    # Check for empty aliases
    for i, alias in enumerate(entity.aliases):
        if not alias or not alias.strip():
            errors.append(ValidationError(
                code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                message=f"Entity '{entity.id}' has empty alias at index {i}",
                path=f"{entity_path}/aliases/{i}",
                severity=ValidationSeverity.WARNING,
            ))

    # Validate date consistency
    if entity.effective_date and entity.expiration_date:
        if entity.effective_date > entity.expiration_date:
            errors.append(ValidationError(
                code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                message=f"Entity '{entity.id}' has effective_date after expiration_date",
                path=f"{entity_path}",
                severity=ValidationSeverity.ERROR,
                details={
                    "effective_date": entity.effective_date.isoformat(),
                    "expiration_date": entity.expiration_date.isoformat(),
                },
            ))

    # Validate deprecation consistency
    if entity.deprecated and not entity.deprecation_info:
        errors.append(ValidationError(
            code=GLNORMErrorCode.E402_ENTITY_DEPRECATED.value,
            message=f"Entity '{entity.id}' is deprecated but missing deprecation info",
            path=f"{entity_path}/deprecation_info",
            severity=ValidationSeverity.INFO,
            suggestion="Provide deprecation_info with reason and optional replacement",
        ))

    return errors


def check_deprecations(
    vocabulary: Vocabulary,
    warn_days_before_removal: int = 30,
) -> List[DeprecationWarning]:
    """
    Check for deprecated entities in a vocabulary.

    Args:
        vocabulary: The vocabulary to check.
        warn_days_before_removal: Warn if removal is within this many days.

    Returns:
        List of DeprecationWarning objects.

    Example:
        >>> warnings = check_deprecations(vocabulary)
        >>> for warning in warnings:
        ...     print(f"Deprecated: {warning.entity_name} - {warning.reason}")
    """
    warnings: List[DeprecationWarning] = []

    for entity in vocabulary.entities.values():
        if not entity.deprecated:
            continue

        warning = DeprecationWarning(
            entity_id=entity.id,
            entity_name=entity.canonical_name,
            deprecated_at=entity.deprecation_info.deprecated_at if entity.deprecation_info else datetime.utcnow(),
            reason=entity.deprecation_info.reason if entity.deprecation_info else "Unknown",
            replacement_id=entity.deprecation_info.replacement_id if entity.deprecation_info else None,
            removal_date=entity.deprecation_info.removal_date if entity.deprecation_info else None,
        )

        warnings.append(warning)

        # Log urgent warnings
        if warning.days_until_removal is not None:
            if warning.days_until_removal <= warn_days_before_removal:
                logger.warning(
                    "Entity will be removed soon",
                    entity_id=entity.id,
                    days_until_removal=warning.days_until_removal,
                    replacement_id=warning.replacement_id,
                )

    logger.info(
        "Deprecation check complete",
        vocab_id=vocabulary.id,
        deprecated_count=len(warnings),
    )

    return warnings


def validate_vocabulary(
    vocabulary: Vocabulary,
    verify_signature: bool = True,
    strict: bool = False,
) -> ValidationResult:
    """
    Perform complete validation of a vocabulary.

    Combines signature verification, schema validation, and deprecation
    checking into a single validation result.

    Args:
        vocabulary: The vocabulary to validate.
        verify_signature: Whether to verify the vocabulary signature.
        strict: If True, treats warnings as errors.

    Returns:
        ValidationResult with all validation findings.

    Example:
        >>> result = validate_vocabulary(vocab, verify_signature=True)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error.message}")
    """
    errors: List[ValidationError] = []
    warnings: List[ValidationError] = []
    info: List[ValidationError] = []

    # Signature validation
    if verify_signature:
        if vocabulary.metadata and vocabulary.metadata.signature:
            if not validate_signature(vocabulary):
                errors.append(ValidationError(
                    code=GLNORMErrorCode.E502_VOCABULARY_CORRUPTED.value,
                    message="Vocabulary signature verification failed",
                    path="/metadata/signature",
                    severity=ValidationSeverity.ERROR,
                    suggestion="Vocabulary may have been tampered with. Obtain a fresh copy from trusted source.",
                ))
        else:
            warnings.append(ValidationError(
                code=GLNORMErrorCode.E504_GOVERNANCE_REQUIRED.value,
                message="Vocabulary has no signature for verification",
                path="/metadata/signature",
                severity=ValidationSeverity.WARNING,
                suggestion="Sign vocabulary with compute_signature() for integrity verification",
            ))

    # Schema validation
    schema_errors = validate_schema(vocabulary, strict=strict)
    for error in schema_errors:
        if error.severity == ValidationSeverity.ERROR:
            errors.append(error)
        elif error.severity == ValidationSeverity.WARNING:
            if strict:
                errors.append(error)
            else:
                warnings.append(error)
        else:
            info.append(error)

    # Deprecation checking
    deprecations = check_deprecations(vocabulary)
    for dep in deprecations:
        severity = ValidationSeverity.WARNING
        if dep.days_until_removal is not None and dep.days_until_removal <= 7:
            severity = ValidationSeverity.ERROR if strict else ValidationSeverity.WARNING

        msg = f"Entity '{dep.entity_name}' is deprecated: {dep.reason}"
        if dep.replacement_id:
            msg += f" (replacement: {dep.replacement_id})"

        warning = ValidationError(
            code=GLNORMErrorCode.E402_ENTITY_DEPRECATED.value,
            message=msg,
            path=f"/entities/{dep.entity_id}",
            severity=severity,
            details=dep.to_dict(),
            suggestion=f"Migrate to replacement entity '{dep.replacement_id}'" if dep.replacement_id else "Remove usage of deprecated entity",
        )
        warnings.append(warning)

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        info=info,
        vocabulary_id=vocabulary.id,
        vocabulary_version=vocabulary.version,
    )

    logger.info(
        "Vocabulary validation complete",
        vocab_id=vocabulary.id,
        is_valid=is_valid,
        error_count=len(errors),
        warning_count=len(warnings),
    )

    return result


__all__ = [
    "ValidationSeverity",
    "ValidationError",
    "DeprecationWarning",
    "ValidationResult",
    "validate_signature",
    "validate_schema",
    "check_deprecations",
    "validate_vocabulary",
]
