"""
Audit payload builder for GL-FOUND-X-003 normalization events.

This module provides a builder pattern for constructing complete audit
payloads from normalization results. It handles:
- Building MeasurementAudit records from conversion results
- Building EntityAudit records from resolution results
- Assembling complete NormalizationEvent records with hash computation

Key Design Principles:
    - Fluent builder API for easy construction
    - Automatic hash computation for integrity
    - Version metadata injection for determinism
    - Thread-safe for concurrent builds

Example:
    >>> from gl_normalizer_core.audit.builder import AuditPayloadBuilder
    >>> builder = AuditPayloadBuilder()
    >>> event = (
    ...     builder
    ...     .set_request_context(request_id="req-123", source_record_id="meter-001")
    ...     .set_org_context(org_id="org-acme", policy_mode="STRICT")
    ...     .set_versions(vocab_version="2026.01.0", ...)
    ...     .add_measurement_audit(measurement_result)
    ...     .add_entity_audit(entity_result)
    ...     .build()
    ... )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .chain import HashChainGenerator, get_default_generator
from .schema import (
    AuditError,
    AuditWarning,
    ConversionStep,
    EntityAudit,
    EventStatus,
    MeasurementAudit,
    NormalizationEvent,
    ResolutionCandidate,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """
    Result of a unit conversion operation.

    This dataclass captures all information needed to build a MeasurementAudit
    record from a conversion operation performed by the conversion engine.

    Attributes:
        field: Name of the field being converted.
        raw_value: Original numeric value.
        raw_unit: Original unit string.
        expected_dimension: Expected dimension from schema.
        parsed_unit_ast: Parsed unit AST as dictionary.
        canonical_value: Converted canonical value.
        canonical_unit: Target canonical unit.
        dimension: Computed dimension.
        conversion_steps: List of conversion steps applied.
        precision_config: Precision configuration applied.
        warnings: List of warning messages.

    Example:
        >>> result = ConversionResult(
        ...     field="energy_consumption",
        ...     raw_value=1500,
        ...     raw_unit="kWh",
        ...     expected_dimension="energy",
        ...     parsed_unit_ast={"normalized_string": "kWh"},
        ...     canonical_value=5400.0,
        ...     canonical_unit="MJ",
        ...     dimension="energy",
        ...     conversion_steps=[...],
        ...     precision_config={"rule": "dimension_default", "digits": 6}
        ... )
    """

    field: str
    raw_value: float
    raw_unit: str
    expected_dimension: str
    parsed_unit_ast: Dict[str, Any]
    canonical_value: float
    canonical_unit: str
    dimension: str
    conversion_steps: List[Dict[str, Any]] = field(default_factory=list)
    precision_config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResolutionResult:
    """
    Result of an entity resolution operation.

    This dataclass captures all information needed to build an EntityAudit
    record from a resolution operation performed by the reference resolver.

    Attributes:
        field: Name of the field being resolved.
        entity_type: Type of entity (fuel, material, process).
        raw_name: Original name from input.
        reference_id: Resolved reference ID.
        canonical_name: Canonical name from vocabulary.
        match_method: Method used for resolution.
        confidence: Confidence score (0.0-1.0).
        vocabulary_version: Version of vocabulary used.
        candidates: Optional list of candidates considered.
        needs_review: Whether human review is required.
        hints_used: Optional hints that influenced resolution.
        warnings: List of warning messages.

    Example:
        >>> result = ResolutionResult(
        ...     field="fuel_type",
        ...     entity_type="fuel",
        ...     raw_name="Nat Gas",
        ...     reference_id="GL-FUEL-NATGAS",
        ...     canonical_name="Natural gas",
        ...     match_method="alias",
        ...     confidence=1.0,
        ...     vocabulary_version="2026.01.0"
        ... )
    """

    field: str
    entity_type: str
    raw_name: str
    reference_id: str
    canonical_name: str
    match_method: str
    confidence: float
    vocabulary_version: str
    candidates: Optional[List[Dict[str, Any]]] = None
    needs_review: bool = False
    hints_used: Optional[Dict[str, str]] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class VersionMetadata:
    """
    Version metadata for determinism tracking.

    All versions must be captured to ensure normalization results
    can be reproduced exactly.

    Attributes:
        vocab_version: Version of controlled vocabulary.
        policy_version: Version of policy configuration.
        unit_registry_version: Version of unit registry.
        validator_version: Version of validator.
        api_revision: API revision of normalizer service.

    Example:
        >>> versions = VersionMetadata(
        ...     vocab_version="2026.01.0",
        ...     policy_version="1.0.0",
        ...     unit_registry_version="2026.01.0",
        ...     validator_version="1.0.0",
        ...     api_revision="v1"
        ... )
    """

    vocab_version: str
    policy_version: str
    unit_registry_version: str
    validator_version: str
    api_revision: str


class AuditPayloadBuilder:
    """
    Builder for constructing NormalizationEvent audit records.

    Provides a fluent API for building complete audit events with
    automatic hash computation and chain linking.

    Thread Safety:
        Each builder instance should be used for a single event.
        The underlying hash chain generator is thread-safe.

    Attributes:
        _hash_generator: Hash chain generator for integrity hashes.
        _request_id: Request ID for correlation.
        _source_record_id: Source record ID being normalized.
        _org_id: Organization ID.
        _policy_mode: Policy mode (STRICT or LENIENT).
        _versions: Version metadata.
        _measurements: List of measurement audit records.
        _entities: List of entity audit records.
        _errors: List of error records.
        _warnings: List of warning records.

    Example:
        >>> builder = AuditPayloadBuilder()
        >>> event = (
        ...     builder
        ...     .set_request_context("req-123", "meter-001")
        ...     .set_org_context("org-acme", "STRICT")
        ...     .set_versions(VersionMetadata(...))
        ...     .add_measurement_audit(conversion_result)
        ...     .add_entity_audit(resolution_result)
        ...     .build()
        ... )
    """

    def __init__(
        self,
        hash_generator: Optional[HashChainGenerator] = None,
    ):
        """
        Initialize the audit payload builder.

        Args:
            hash_generator: Optional custom hash generator.
                Uses default singleton if not provided.
        """
        self._hash_generator = hash_generator or get_default_generator()
        self._reset()

    def _reset(self) -> None:
        """Reset builder state for reuse."""
        self._event_id: Optional[str] = None
        self._request_id: Optional[str] = None
        self._source_record_id: Optional[str] = None
        self._org_id: Optional[str] = None
        self._policy_mode: str = "STRICT"
        self._versions: Optional[VersionMetadata] = None
        self._measurements: List[MeasurementAudit] = []
        self._entities: List[EntityAudit] = []
        self._errors: List[Union[AuditError, Dict[str, Any]]] = []
        self._warnings: List[Union[AuditWarning, Dict[str, Any]]] = []
        self._status: Optional[EventStatus] = None
        self._prev_event_hash: Optional[str] = None

    def set_event_id(self, event_id: str) -> "AuditPayloadBuilder":
        """
        Set a custom event ID.

        If not set, an ID will be auto-generated during build().

        Args:
            event_id: Custom event ID.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_event_id("norm-evt-custom-001")
        """
        self._event_id = event_id
        return self

    def set_request_context(
        self,
        request_id: str,
        source_record_id: str,
    ) -> "AuditPayloadBuilder":
        """
        Set request context for the audit event.

        Args:
            request_id: Correlation ID for the normalization request.
            source_record_id: ID of the source record being normalized.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_request_context("req-123", "meter-2026-001")
        """
        self._request_id = request_id
        self._source_record_id = source_record_id
        return self

    def set_org_context(
        self,
        org_id: str,
        policy_mode: str = "STRICT",
    ) -> "AuditPayloadBuilder":
        """
        Set organization context for the audit event.

        Args:
            org_id: Organization ID for multi-tenant environments.
            policy_mode: Policy mode (STRICT or LENIENT).

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_org_context("org-acme", "STRICT")
        """
        self._org_id = org_id
        self._policy_mode = policy_mode.upper()
        return self

    def set_versions(
        self,
        versions: Optional[VersionMetadata] = None,
        *,
        vocab_version: Optional[str] = None,
        policy_version: Optional[str] = None,
        unit_registry_version: Optional[str] = None,
        validator_version: Optional[str] = None,
        api_revision: Optional[str] = None,
    ) -> "AuditPayloadBuilder":
        """
        Set version metadata for determinism tracking.

        Can be called with a VersionMetadata object or individual parameters.

        Args:
            versions: VersionMetadata object (takes precedence).
            vocab_version: Version of controlled vocabulary.
            policy_version: Version of policy configuration.
            unit_registry_version: Version of unit registry.
            validator_version: Version of validator.
            api_revision: API revision of normalizer service.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_versions(
            ...     vocab_version="2026.01.0",
            ...     policy_version="1.0.0",
            ...     unit_registry_version="2026.01.0",
            ...     validator_version="1.0.0",
            ...     api_revision="v1"
            ... )
        """
        if versions is not None:
            self._versions = versions
        else:
            self._versions = VersionMetadata(
                vocab_version=vocab_version or "unknown",
                policy_version=policy_version or "unknown",
                unit_registry_version=unit_registry_version or "unknown",
                validator_version=validator_version or "unknown",
                api_revision=api_revision or "unknown",
            )
        return self

    def set_prev_event_hash(self, prev_event_hash: Optional[str]) -> "AuditPayloadBuilder":
        """
        Set the previous event hash for chain linking.

        If not set, the builder will attempt to get it from the chain
        state during build().

        Args:
            prev_event_hash: Hash of the previous event in the chain.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_prev_event_hash("sha256:abc123...")
        """
        self._prev_event_hash = prev_event_hash
        return self

    def add_measurement_audit(
        self,
        result: Union[ConversionResult, Dict[str, Any]],
    ) -> "AuditPayloadBuilder":
        """
        Add a measurement audit record from a conversion result.

        Args:
            result: ConversionResult or dictionary with conversion data.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_measurement_audit(conversion_result)
        """
        if isinstance(result, ConversionResult):
            audit = self._build_measurement_audit_from_result(result)
        elif isinstance(result, dict):
            audit = self._build_measurement_audit_from_dict(result)
        else:
            raise TypeError(
                f"Expected ConversionResult or dict, got {type(result).__name__}"
            )

        self._measurements.append(audit)
        logger.debug(
            "Added measurement audit for field=%s (raw=%s %s -> %s %s)",
            audit.field,
            audit.raw_value,
            audit.raw_unit,
            audit.canonical_value,
            audit.canonical_unit,
        )
        return self

    def add_entity_audit(
        self,
        result: Union[ResolutionResult, Dict[str, Any]],
    ) -> "AuditPayloadBuilder":
        """
        Add an entity audit record from a resolution result.

        Args:
            result: ResolutionResult or dictionary with resolution data.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_entity_audit(resolution_result)
        """
        if isinstance(result, ResolutionResult):
            audit = self._build_entity_audit_from_result(result)
        elif isinstance(result, dict):
            audit = self._build_entity_audit_from_dict(result)
        else:
            raise TypeError(
                f"Expected ResolutionResult or dict, got {type(result).__name__}"
            )

        self._entities.append(audit)
        logger.debug(
            "Added entity audit for field=%s (raw=%s -> %s, method=%s)",
            audit.field,
            audit.raw_name,
            audit.reference_id,
            audit.match_method,
        )
        return self

    def add_error(
        self,
        error: Union[AuditError, Dict[str, Any]],
    ) -> "AuditPayloadBuilder":
        """
        Add an error record to the audit event.

        Args:
            error: AuditError or dictionary with error data.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_error(AuditError(
            ...     code="GLNORM-E200",
            ...     severity="error",
            ...     path="/measurements/0",
            ...     message="Dimension mismatch"
            ... ))
        """
        self._errors.append(error)
        code = error.code if isinstance(error, AuditError) else error.get("code", "unknown")
        logger.debug("Added error: %s", code)
        return self

    def add_warning(
        self,
        warning: Union[AuditWarning, Dict[str, Any]],
    ) -> "AuditPayloadBuilder":
        """
        Add a warning record to the audit event.

        Args:
            warning: AuditWarning or dictionary with warning data.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.add_warning(AuditWarning(
            ...     code="GLNORM-E403",
            ...     path="/entities/0",
            ...     message="Low confidence match (0.85)"
            ... ))
        """
        self._warnings.append(warning)
        code = warning.code if isinstance(warning, AuditWarning) else warning.get("code", "unknown")
        logger.debug("Added warning: %s", code)
        return self

    def set_status(self, status: EventStatus) -> "AuditPayloadBuilder":
        """
        Explicitly set the event status.

        If not set, status is automatically determined from errors/warnings.

        Args:
            status: Event status to set.

        Returns:
            Self for method chaining.

        Example:
            >>> builder.set_status(EventStatus.WARNING)
        """
        self._status = status
        return self

    def build(self) -> NormalizationEvent:
        """
        Build the complete NormalizationEvent.

        Performs validation, computes hashes, and creates the immutable
        audit event record.

        Returns:
            Complete NormalizationEvent with computed hashes.

        Raises:
            ValueError: If required fields are not set.

        Example:
            >>> event = builder.build()
            >>> print(event.event_id)
            >>> print(event.event_hash)
        """
        # Validate required fields
        self._validate_required_fields()

        # Generate event ID if not set
        event_id = self._event_id or self._hash_generator.generate_event_id()

        # Determine status
        status = self._determine_status()

        # Compute payload hash
        payload_hash = self._hash_generator.compute_payload_hash(
            measurements=[m.model_dump(mode="json") for m in self._measurements],
            entities=[e.model_dump(mode="json") for e in self._entities],
        )

        # Get prev_event_hash from chain if not set
        prev_event_hash = self._prev_event_hash
        if prev_event_hash is None and self._org_id:
            prev_event_hash = self._hash_generator.get_prev_event_hash(self._org_id)

        # Create event timestamp
        event_ts = datetime.utcnow()

        # Build event data for hash computation
        event_data = {
            "event_id": event_id,
            "event_ts": event_ts.isoformat() + "Z",
            "prev_event_hash": prev_event_hash,
            "request_id": self._request_id,
            "source_record_id": self._source_record_id,
            "org_id": self._org_id,
            "policy_mode": self._policy_mode,
            "status": status.value,
            "vocab_version": self._versions.vocab_version,
            "policy_version": self._versions.policy_version,
            "unit_registry_version": self._versions.unit_registry_version,
            "validator_version": self._versions.validator_version,
            "api_revision": self._versions.api_revision,
            "measurements": [m.model_dump(mode="json") for m in self._measurements],
            "entities": [e.model_dump(mode="json") for e in self._entities],
            "errors": self._serialize_errors(),
            "warnings": self._serialize_warnings(),
            "payload_hash": payload_hash,
        }

        # Compute event hash
        event_hash = self._hash_generator.compute_event_hash(event_data, prev_event_hash)

        # Link event to chain
        if self._org_id:
            self._hash_generator.link_event(self._org_id, event_id, event_hash)

        # Create the event
        event = NormalizationEvent(
            event_id=event_id,
            event_ts=event_ts,
            prev_event_hash=prev_event_hash,
            request_id=self._request_id,
            source_record_id=self._source_record_id,
            org_id=self._org_id,
            policy_mode=self._policy_mode,
            status=status,
            vocab_version=self._versions.vocab_version,
            policy_version=self._versions.policy_version,
            unit_registry_version=self._versions.unit_registry_version,
            validator_version=self._versions.validator_version,
            api_revision=self._versions.api_revision,
            measurements=self._measurements,
            entities=self._entities,
            errors=self._errors,
            warnings=self._warnings,
            payload_hash=payload_hash,
            event_hash=event_hash,
        )

        logger.info(
            "Built audit event: event_id=%s, status=%s, measurements=%d, entities=%d",
            event_id,
            status.value,
            len(self._measurements),
            len(self._entities),
        )

        # Reset builder for reuse
        self._reset()

        return event

    def _validate_required_fields(self) -> None:
        """Validate that all required fields are set."""
        missing = []

        if not self._request_id:
            missing.append("request_id")
        if not self._source_record_id:
            missing.append("source_record_id")
        if not self._org_id:
            missing.append("org_id")
        if not self._versions:
            missing.append("versions")

        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}. "
                f"Use set_request_context(), set_org_context(), and set_versions() "
                f"before calling build()."
            )

    def _determine_status(self) -> EventStatus:
        """Determine the event status from errors and warnings."""
        if self._status is not None:
            return self._status

        if self._errors:
            return EventStatus.FAILED

        if self._warnings:
            return EventStatus.WARNING

        # Check if any entity needs review
        for entity in self._entities:
            if entity.needs_review:
                return EventStatus.WARNING

        return EventStatus.SUCCESS

    def _build_measurement_audit_from_result(
        self,
        result: ConversionResult,
    ) -> MeasurementAudit:
        """Build MeasurementAudit from ConversionResult."""
        conversion_steps = [
            ConversionStep(**step) if isinstance(step, dict) else step
            for step in result.conversion_steps
        ]

        return MeasurementAudit(
            field=result.field,
            raw_value=result.raw_value,
            raw_unit=result.raw_unit,
            expected_dimension=result.expected_dimension,
            parsed_unit_ast=result.parsed_unit_ast,
            canonical_value=result.canonical_value,
            canonical_unit=result.canonical_unit,
            dimension=result.dimension,
            conversion_steps=conversion_steps,
            precision_applied=result.precision_config,
            warnings=result.warnings,
        )

    def _build_measurement_audit_from_dict(
        self,
        data: Dict[str, Any],
    ) -> MeasurementAudit:
        """Build MeasurementAudit from dictionary."""
        # Handle nested conversion_steps
        conversion_steps = data.get("conversion_steps", [])
        if conversion_steps:
            conversion_steps = [
                ConversionStep(**step) if isinstance(step, dict) else step
                for step in conversion_steps
            ]

        return MeasurementAudit(
            field=data["field"],
            raw_value=data["raw_value"],
            raw_unit=data["raw_unit"],
            expected_dimension=data["expected_dimension"],
            parsed_unit_ast=data.get("parsed_unit_ast", {}),
            canonical_value=data["canonical_value"],
            canonical_unit=data["canonical_unit"],
            dimension=data["dimension"],
            conversion_steps=conversion_steps,
            precision_applied=data.get("precision_applied", {}),
            warnings=data.get("warnings", []),
        )

    def _build_entity_audit_from_result(
        self,
        result: ResolutionResult,
    ) -> EntityAudit:
        """Build EntityAudit from ResolutionResult."""
        candidates = None
        if result.candidates:
            candidates = [
                ResolutionCandidate(**c) if isinstance(c, dict) else c
                for c in result.candidates
            ]

        return EntityAudit(
            field=result.field,
            entity_type=result.entity_type,
            raw_name=result.raw_name,
            reference_id=result.reference_id,
            canonical_name=result.canonical_name,
            match_method=result.match_method,
            confidence=result.confidence,
            vocabulary_version=result.vocabulary_version,
            candidates_considered=candidates,
            needs_review=result.needs_review,
            hints_used=result.hints_used,
            warnings=result.warnings,
        )

    def _build_entity_audit_from_dict(
        self,
        data: Dict[str, Any],
    ) -> EntityAudit:
        """Build EntityAudit from dictionary."""
        candidates = data.get("candidates_considered")
        if candidates:
            candidates = [
                ResolutionCandidate(**c) if isinstance(c, dict) else c
                for c in candidates
            ]

        return EntityAudit(
            field=data["field"],
            entity_type=data["entity_type"],
            raw_name=data["raw_name"],
            reference_id=data["reference_id"],
            canonical_name=data["canonical_name"],
            match_method=data["match_method"],
            confidence=data["confidence"],
            vocabulary_version=data["vocabulary_version"],
            candidates_considered=candidates,
            needs_review=data.get("needs_review", False),
            hints_used=data.get("hints_used"),
            warnings=data.get("warnings", []),
        )

    def _serialize_errors(self) -> List[Dict[str, Any]]:
        """Serialize errors to dictionaries."""
        return [
            e.model_dump(mode="json") if isinstance(e, AuditError) else e
            for e in self._errors
        ]

    def _serialize_warnings(self) -> List[Dict[str, Any]]:
        """Serialize warnings to dictionaries."""
        return [
            w.model_dump(mode="json") if isinstance(w, AuditWarning) else w
            for w in self._warnings
        ]


def build_measurement_audit(
    field: str,
    raw_value: float,
    raw_unit: str,
    expected_dimension: str,
    canonical_value: float,
    canonical_unit: str,
    dimension: str,
    conversion_steps: Optional[List[Dict[str, Any]]] = None,
    parsed_unit_ast: Optional[Dict[str, Any]] = None,
    precision_config: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> MeasurementAudit:
    """
    Convenience function to build a MeasurementAudit directly.

    Args:
        field: Name of the field.
        raw_value: Original numeric value.
        raw_unit: Original unit string.
        expected_dimension: Expected dimension from schema.
        canonical_value: Converted canonical value.
        canonical_unit: Target canonical unit.
        dimension: Computed dimension.
        conversion_steps: List of conversion step dictionaries.
        parsed_unit_ast: Parsed unit AST.
        precision_config: Precision configuration.
        warnings: List of warning messages.

    Returns:
        MeasurementAudit record.

    Example:
        >>> audit = build_measurement_audit(
        ...     field="energy",
        ...     raw_value=100,
        ...     raw_unit="kWh",
        ...     expected_dimension="energy",
        ...     canonical_value=360,
        ...     canonical_unit="MJ",
        ...     dimension="energy"
        ... )
    """
    steps = []
    if conversion_steps:
        steps = [
            ConversionStep(**s) if isinstance(s, dict) else s
            for s in conversion_steps
        ]

    return MeasurementAudit(
        field=field,
        raw_value=raw_value,
        raw_unit=raw_unit,
        expected_dimension=expected_dimension,
        parsed_unit_ast=parsed_unit_ast or {},
        canonical_value=canonical_value,
        canonical_unit=canonical_unit,
        dimension=dimension,
        conversion_steps=steps,
        precision_applied=precision_config or {},
        warnings=warnings or [],
    )


def build_entity_audit(
    field: str,
    entity_type: str,
    raw_name: str,
    reference_id: str,
    canonical_name: str,
    match_method: str,
    confidence: float,
    vocabulary_version: str,
    needs_review: bool = False,
    candidates: Optional[List[Dict[str, Any]]] = None,
    hints_used: Optional[Dict[str, str]] = None,
    warnings: Optional[List[str]] = None,
) -> EntityAudit:
    """
    Convenience function to build an EntityAudit directly.

    Args:
        field: Name of the field.
        entity_type: Type of entity (fuel, material, process).
        raw_name: Original name from input.
        reference_id: Resolved reference ID.
        canonical_name: Canonical name from vocabulary.
        match_method: Method used for resolution.
        confidence: Confidence score (0.0-1.0).
        vocabulary_version: Version of vocabulary used.
        needs_review: Whether human review is required.
        candidates: List of candidate dictionaries.
        hints_used: Hints that influenced resolution.
        warnings: List of warning messages.

    Returns:
        EntityAudit record.

    Example:
        >>> audit = build_entity_audit(
        ...     field="fuel_type",
        ...     entity_type="fuel",
        ...     raw_name="Nat Gas",
        ...     reference_id="GL-FUEL-NATGAS",
        ...     canonical_name="Natural gas",
        ...     match_method="alias",
        ...     confidence=1.0,
        ...     vocabulary_version="2026.01.0"
        ... )
    """
    candidate_list = None
    if candidates:
        candidate_list = [
            ResolutionCandidate(**c) if isinstance(c, dict) else c
            for c in candidates
        ]

    return EntityAudit(
        field=field,
        entity_type=entity_type,
        raw_name=raw_name,
        reference_id=reference_id,
        canonical_name=canonical_name,
        match_method=match_method,
        confidence=confidence,
        vocabulary_version=vocabulary_version,
        candidates_considered=candidate_list,
        needs_review=needs_review,
        hints_used=hints_used,
        warnings=warnings or [],
    )
