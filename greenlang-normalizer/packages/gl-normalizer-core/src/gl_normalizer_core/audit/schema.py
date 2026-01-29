"""
Audit event schemas for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the Pydantic models for the complete audit event system,
including measurement audits, entity audits, and the full normalization event
envelope. These schemas support governance-grade audit trails with full
lineage from raw input to canonical output.

Key Design Principles:
    - Immutable audit records with hash chaining for tamper detection
    - Complete conversion traces for reproducibility
    - Version tracking for all reference data used
    - Structured error and warning payloads

Example:
    >>> from gl_normalizer_core.audit.schema import NormalizationEvent, EventStatus
    >>> event = NormalizationEvent(
    ...     event_id="norm-evt-001",
    ...     event_ts=datetime.utcnow(),
    ...     request_id="req-123",
    ...     source_record_id="meter-001",
    ...     org_id="org-acme",
    ...     policy_mode="STRICT",
    ...     status=EventStatus.SUCCESS,
    ...     # ... other fields
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class EventStatus(str, Enum):
    """
    Status of a normalization event.

    Attributes:
        SUCCESS: All operations completed successfully without warnings.
        WARNING: Operations completed but with warnings (e.g., low confidence match).
        FAILED: One or more operations failed; partial results may be available.
    """

    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


class ConversionMethod(str, Enum):
    """
    Method used for a unit conversion step.

    Attributes:
        MULTIPLY: Simple multiplication by conversion factor.
        DIVIDE: Division by conversion factor.
        AFFINE: Affine transformation (offset + scale) for temperature conversions.
        BASIS_CONVERSION: Basis-dependent conversion requiring reference conditions.
        COMPOUND: Multi-step compound unit conversion.
    """

    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AFFINE = "affine"
    BASIS_CONVERSION = "basis_conversion"
    COMPOUND = "compound"


class MatchMethod(str, Enum):
    """
    Method used for entity resolution.

    Ordered by determinism and confidence level (highest to lowest).

    Attributes:
        EXACT_ID: Direct reference ID match.
        EXACT_NAME: Exact canonical name match (case-normalized).
        ALIAS: Exact alias match from vocabulary.
        RULE: Rule-based normalization match (e.g., hyphen removal).
        FUZZY: Non-deterministic fuzzy matching (token overlap, edit distance).
        LLM_CANDIDATE: LLM-suggested candidate requiring human review.
    """

    EXACT_ID = "exact_id"
    EXACT_NAME = "exact_name"
    ALIAS = "alias"
    RULE = "rule"
    FUZZY = "fuzzy"
    LLM_CANDIDATE = "llm_candidate"


class EntityType(str, Enum):
    """
    Type of entity being resolved.

    Attributes:
        FUEL: Fuel type entity (e.g., natural gas, diesel).
        MATERIAL: Material entity (e.g., Portland cement, steel).
        PROCESS: Process entity (e.g., electric arc furnace).
    """

    FUEL = "fuel"
    MATERIAL = "material"
    PROCESS = "process"


class ReferenceConditions(BaseModel):
    """
    Reference conditions for basis-dependent unit conversions.

    Required for conversions involving normal cubic meters (Nm3),
    standard cubic feet (scf), and similar volume-based units.

    Attributes:
        temperature_c: Reference temperature in degrees Celsius.
        pressure_kpa: Reference pressure in kilopascals.

    Example:
        >>> conditions = ReferenceConditions(temperature_c=0, pressure_kpa=101.325)
    """

    temperature_c: float = Field(
        ...,
        description="Reference temperature in degrees Celsius",
        alias="temperature_C",
    )
    pressure_kpa: float = Field(
        default=101.325,
        ge=0,
        description="Reference pressure in kilopascals",
        alias="pressure_kPa",
    )

    model_config = {"populate_by_name": True}


class ConversionStep(BaseModel):
    """
    Single step in a unit conversion trace.

    Captures all metadata required to reproduce the conversion, including
    the conversion factor, method, and version of the factor used.

    Attributes:
        from_unit: Source unit for this step.
        to_unit: Target unit for this step.
        factor: Conversion factor applied.
        method: Conversion method used (multiply, divide, affine, etc.).
        factor_version: Version of the conversion factor registry used.
        reference_conditions: Optional reference conditions for basis conversions.
        offset: Optional offset for affine transformations (temperature).
        intermediate_value: Optional intermediate value after this step.

    Example:
        >>> step = ConversionStep(
        ...     from_unit="kWh",
        ...     to_unit="MJ",
        ...     factor=3.6,
        ...     method="multiply",
        ...     factor_version="2026.01.0"
        ... )
    """

    from_unit: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Source unit for this conversion step",
    )
    to_unit: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target unit for this conversion step",
    )
    factor: float = Field(
        ...,
        description="Conversion factor applied in this step",
    )
    method: str = Field(
        ...,
        description="Conversion method: multiply, divide, affine, basis_conversion, compound",
    )
    factor_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of the conversion factor registry used",
    )
    reference_conditions: Optional[ReferenceConditions] = Field(
        default=None,
        description="Reference conditions for basis-dependent conversions",
    )
    offset: Optional[float] = Field(
        default=None,
        description="Offset for affine transformations (e.g., temperature)",
    )
    intermediate_value: Optional[float] = Field(
        default=None,
        description="Value after applying this conversion step",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate conversion method is one of the allowed values."""
        allowed_methods = {"multiply", "divide", "affine", "basis_conversion", "compound"}
        if v.lower() not in allowed_methods:
            raise ValueError(
                f"Invalid conversion method '{v}'. "
                f"Allowed methods: {', '.join(sorted(allowed_methods))}"
            )
        return v.lower()


class PrecisionConfig(BaseModel):
    """
    Precision configuration applied to a conversion result.

    Attributes:
        rule: Name of the precision rule applied.
        digits: Number of significant digits preserved.
        rounding_mode: Rounding mode used (e.g., HALF_EVEN).
        original_value: Value before precision rounding.
        rounded_value: Value after precision rounding.

    Example:
        >>> precision = PrecisionConfig(
        ...     rule="dimension_default",
        ...     digits=6,
        ...     rounding_mode="HALF_EVEN"
        ... )
    """

    rule: str = Field(
        ...,
        description="Name of the precision rule applied",
    )
    digits: int = Field(
        ...,
        ge=1,
        le=15,
        description="Number of significant digits preserved",
    )
    rounding_mode: str = Field(
        default="HALF_EVEN",
        description="Rounding mode used",
    )
    original_value: Optional[float] = Field(
        default=None,
        description="Value before precision rounding",
    )
    rounded_value: Optional[float] = Field(
        default=None,
        description="Value after precision rounding",
    )


class UnitAST(BaseModel):
    """
    Abstract Syntax Tree representation of a parsed unit.

    Captures the structural decomposition of a unit string into
    numerator and denominator terms with prefixes and exponents.

    Attributes:
        normalized_string: Normalized form of the unit string.
        numerator_terms: List of terms in the numerator.
        denominator_terms: List of terms in the denominator.
        dimension_signature: Computed dimension signature.

    Example:
        >>> ast = UnitAST(
        ...     normalized_string="kg/m3",
        ...     numerator_terms=[{"symbol": "kg", "prefix": "k", "exponent": 1}],
        ...     denominator_terms=[{"symbol": "m", "prefix": None, "exponent": 3}],
        ...     dimension_signature="mass/volume"
        ... )
    """

    normalized_string: str = Field(
        ...,
        description="Normalized form of the unit string",
    )
    numerator_terms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of terms in the numerator",
    )
    denominator_terms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of terms in the denominator",
    )
    dimension_signature: Optional[str] = Field(
        default=None,
        description="Computed dimension signature",
    )


class MeasurementAudit(BaseModel):
    """
    Audit record for a single measurement normalization.

    Contains complete lineage from raw input to canonical output,
    including all conversion steps, precision rules, and warnings.

    Attributes:
        field: Name of the field being normalized.
        raw_value: Original numeric value from input.
        raw_unit: Original unit string from input.
        expected_dimension: Expected dimension from schema context.
        parsed_unit_ast: Parsed unit AST representation.
        canonical_value: Converted value in canonical units.
        canonical_unit: Canonical unit for this dimension.
        dimension: Computed dimension of the measurement.
        conversion_steps: List of conversion steps applied.
        precision_applied: Precision configuration applied.
        warnings: List of warning messages.

    Example:
        >>> audit = MeasurementAudit(
        ...     field="energy_consumption",
        ...     raw_value=1500,
        ...     raw_unit="kWh",
        ...     expected_dimension="energy",
        ...     parsed_unit_ast={"normalized_string": "kWh", ...},
        ...     canonical_value=5400.0,
        ...     canonical_unit="MJ",
        ...     dimension="energy",
        ...     conversion_steps=[...],
        ...     precision_applied={"rule": "dimension_default", "digits": 6},
        ...     warnings=[]
        ... )
    """

    field: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the field being normalized",
    )
    raw_value: float = Field(
        ...,
        description="Original numeric value from input",
    )
    raw_unit: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Original unit string from input",
    )
    expected_dimension: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Expected dimension from schema context",
    )
    parsed_unit_ast: Dict[str, Any] = Field(
        ...,
        description="Parsed unit AST representation as dictionary",
    )
    canonical_value: float = Field(
        ...,
        description="Converted value in canonical units",
    )
    canonical_unit: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Canonical unit for this dimension",
    )
    dimension: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Computed dimension of the measurement",
    )
    conversion_steps: List[ConversionStep] = Field(
        default_factory=list,
        description="List of conversion steps applied",
    )
    precision_applied: Dict[str, Any] = Field(
        ...,
        description="Precision configuration applied",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )


class ResolutionCandidate(BaseModel):
    """
    Candidate entity considered during resolution.

    Used for audit trails and review workflows when multiple
    candidates are found or when low-confidence matches are returned.

    Attributes:
        reference_id: Reference ID of the candidate.
        canonical_name: Canonical name of the candidate.
        score: Matching score (0.0-1.0).
        match_method: Method that produced this candidate.
        reasons: List of reasons for the score.

    Example:
        >>> candidate = ResolutionCandidate(
        ...     reference_id="GL-FUEL-NATGAS",
        ...     canonical_name="Natural gas",
        ...     score=0.98,
        ...     match_method="alias"
        ... )
    """

    reference_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Reference ID of the candidate",
    )
    canonical_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Canonical name of the candidate",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Matching score (0.0-1.0)",
    )
    match_method: str = Field(
        ...,
        description="Method that produced this candidate",
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="List of reasons contributing to the score",
    )


class EntityAudit(BaseModel):
    """
    Audit record for a single entity resolution.

    Contains complete lineage from raw input to resolved reference ID,
    including match method, confidence, vocabulary version, and candidates.

    Attributes:
        field: Name of the field being resolved.
        entity_type: Type of entity (fuel, material, process).
        raw_name: Original name from input.
        reference_id: Resolved reference ID.
        canonical_name: Canonical name from vocabulary.
        match_method: Method used for resolution.
        confidence: Confidence score (0.0-1.0).
        vocabulary_version: Version of vocabulary used.
        candidates_considered: Optional list of candidates evaluated.
        needs_review: Whether human review is required.
        hints_used: Optional hints that influenced resolution.
        warnings: List of warning messages.

    Example:
        >>> audit = EntityAudit(
        ...     field="fuel_type",
        ...     entity_type="fuel",
        ...     raw_name="Nat Gas",
        ...     reference_id="GL-FUEL-NATGAS",
        ...     canonical_name="Natural gas",
        ...     match_method="alias",
        ...     confidence=1.0,
        ...     vocabulary_version="2026.01.0",
        ...     needs_review=False
        ... )
    """

    field: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the field being resolved",
    )
    entity_type: str = Field(
        ...,
        description="Type of entity: fuel, material, or process",
    )
    raw_name: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Original name from input",
    )
    reference_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Resolved reference ID",
    )
    canonical_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Canonical name from vocabulary",
    )
    match_method: str = Field(
        ...,
        description="Method used for resolution: exact, alias, rule, fuzzy, llm_candidate",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)",
    )
    vocabulary_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of vocabulary used for resolution",
    )
    candidates_considered: Optional[List[ResolutionCandidate]] = Field(
        default=None,
        description="List of candidates evaluated during resolution",
    )
    needs_review: bool = Field(
        default=False,
        description="Whether human review is required for this resolution",
    )
    hints_used: Optional[Dict[str, str]] = Field(
        default=None,
        description="Hints that influenced resolution (region, sector, etc.)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate entity type is one of the allowed values."""
        allowed_types = {"fuel", "material", "process"}
        if v.lower() not in allowed_types:
            raise ValueError(
                f"Invalid entity type '{v}'. "
                f"Allowed types: {', '.join(sorted(allowed_types))}"
            )
        return v.lower()

    @field_validator("match_method")
    @classmethod
    def validate_match_method(cls, v: str) -> str:
        """Validate match method is one of the allowed values."""
        allowed_methods = {"exact_id", "exact_name", "exact", "alias", "rule", "fuzzy", "llm_candidate"}
        if v.lower() not in allowed_methods:
            raise ValueError(
                f"Invalid match method '{v}'. "
                f"Allowed methods: {', '.join(sorted(allowed_methods))}"
            )
        return v.lower()


class AuditError(BaseModel):
    """
    Structured error record for audit events.

    Follows the GLNORM-Exxx error code taxonomy.

    Attributes:
        code: Error code (e.g., GLNORM-E200).
        severity: Severity level (error, warning, info).
        path: JSON path to the affected field.
        message: Human-readable error message.
        expected: Expected value or constraint.
        actual: Actual value that caused the error.
        hint: Optional remediation hint.

    Example:
        >>> error = AuditError(
        ...     code="GLNORM-E200",
        ...     severity="error",
        ...     path="/measurements/0",
        ...     message="Dimension mismatch: expected 'energy', got 'mass'"
        ... )
    """

    code: str = Field(
        ...,
        pattern=r"^GLNORM-E\d{3}$",
        description="Error code following GLNORM-Exxx format",
    )
    severity: str = Field(
        ...,
        description="Severity level: error, warning, or info",
    )
    path: str = Field(
        ...,
        description="JSON path to the affected field",
    )
    message: str = Field(
        ...,
        max_length=1000,
        description="Human-readable error message",
    )
    expected: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Expected value or constraint",
    )
    actual: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Actual value that caused the error",
    )
    hint: Optional[Dict[str, str]] = Field(
        default=None,
        description="Remediation hint with suggestion and docs link",
    )


class AuditWarning(BaseModel):
    """
    Structured warning record for audit events.

    Attributes:
        code: Warning code (e.g., GLNORM-E307).
        path: JSON path to the affected field.
        message: Human-readable warning message.
        context: Additional context about the warning.

    Example:
        >>> warning = AuditWarning(
        ...     code="GLNORM-E403",
        ...     path="/entities/0",
        ...     message="Low confidence match (0.85)"
        ... )
    """

    code: str = Field(
        ...,
        pattern=r"^GLNORM-E\d{3}$",
        description="Warning code following GLNORM-Exxx format",
    )
    path: str = Field(
        ...,
        description="JSON path to the affected field",
    )
    message: str = Field(
        ...,
        max_length=1000,
        description="Human-readable warning message",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the warning",
    )


class NormalizationEvent(BaseModel):
    """
    Complete audit event for a normalization request.

    This is the top-level audit record that captures all transformations
    applied to a single normalization request. It includes:
    - Complete version tracking for determinism
    - All measurement and entity audits
    - Error and warning records
    - Hash chain for tamper detection

    Attributes:
        event_id: Unique identifier for this audit event.
        event_ts: Timestamp when the event was created.
        prev_event_hash: Hash of the previous event in the chain.
        request_id: Correlation ID for the normalization request.
        source_record_id: ID of the source record being normalized.
        org_id: Organization ID for multi-tenant environments.
        policy_mode: Policy mode used (STRICT or LENIENT).
        status: Final status of the normalization (success/warning/failed).
        vocab_version: Version of the controlled vocabulary used.
        policy_version: Version of the policy configuration used.
        unit_registry_version: Version of the unit registry used.
        validator_version: Version of the validator used.
        api_revision: API revision of the normalizer service.
        measurements: List of measurement audit records.
        entities: List of entity audit records.
        errors: List of error records.
        warnings: List of warning records.
        payload_hash: SHA-256 hash of the payload (measurements + entities).
        event_hash: SHA-256 hash of the complete event including prev_event_hash.

    Example:
        >>> event = NormalizationEvent(
        ...     event_id="norm-evt-abc123",
        ...     event_ts=datetime.utcnow(),
        ...     request_id="req-456",
        ...     source_record_id="meter-2026-001",
        ...     org_id="org-acme",
        ...     policy_mode="STRICT",
        ...     status=EventStatus.SUCCESS,
        ...     vocab_version="2026.01.0",
        ...     policy_version="1.0.0",
        ...     unit_registry_version="2026.01.0",
        ...     validator_version="1.0.0",
        ...     api_revision="v1",
        ...     measurements=[...],
        ...     entities=[...],
        ...     errors=[],
        ...     warnings=[],
        ...     payload_hash="sha256:...",
        ...     event_hash="sha256:..."
        ... )
    """

    # Event identification and linking
    event_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for this audit event",
    )
    event_ts: datetime = Field(
        ...,
        description="Timestamp when the event was created (UTC)",
    )
    prev_event_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the previous event in the chain",
    )

    # Request context
    request_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Correlation ID for the normalization request",
    )
    source_record_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="ID of the source record being normalized",
    )
    org_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Organization ID for multi-tenant environments",
    )
    policy_mode: str = Field(
        ...,
        description="Policy mode used: STRICT or LENIENT",
    )
    status: EventStatus = Field(
        ...,
        description="Final status of the normalization",
    )

    # Version tracking for determinism
    vocab_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of the controlled vocabulary used",
    )
    policy_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of the policy configuration used",
    )
    unit_registry_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of the unit registry used",
    )
    validator_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Version of the validator used",
    )
    api_revision: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="API revision of the normalizer service",
    )

    # Payloads
    measurements: List[MeasurementAudit] = Field(
        default_factory=list,
        description="List of measurement audit records",
    )
    entities: List[EntityAudit] = Field(
        default_factory=list,
        description="List of entity audit records",
    )
    errors: List[Union[AuditError, Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of error records",
    )
    warnings: List[Union[AuditWarning, Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of warning records",
    )

    # Integrity hashes
    payload_hash: str = Field(
        ...,
        description="SHA-256 hash of the payload (measurements + entities)",
    )
    event_hash: str = Field(
        ...,
        description="SHA-256 hash of the complete event including prev_event_hash",
    )

    @field_validator("policy_mode")
    @classmethod
    def validate_policy_mode(cls, v: str) -> str:
        """Validate policy mode is one of the allowed values."""
        allowed_modes = {"STRICT", "LENIENT"}
        if v.upper() not in allowed_modes:
            raise ValueError(
                f"Invalid policy mode '{v}'. "
                f"Allowed modes: {', '.join(sorted(allowed_modes))}"
            )
        return v.upper()

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "NormalizationEvent":
        """Validate status is consistent with errors and warnings."""
        if self.status == EventStatus.FAILED and not self.errors:
            # Allow failed status without errors for edge cases
            pass
        if self.status == EventStatus.SUCCESS and self.errors:
            raise ValueError(
                "Status is SUCCESS but errors list is not empty. "
                "Use WARNING or FAILED status when errors are present."
            )
        return self

    def model_dump_json_stable(self) -> str:
        """
        Serialize to JSON with deterministic key ordering.

        Returns:
            JSON string with sorted keys for reproducible hashing.
        """
        import json
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


# Type aliases for convenience
AuditEventType = NormalizationEvent
ConversionStepType = ConversionStep
MeasurementAuditType = MeasurementAudit
EntityAuditType = EntityAudit
