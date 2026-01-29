"""
GL-FOUND-X-003: GreenLang Normalizer SDK - Data Models

This module defines the Pydantic models used by the GreenLang Normalizer SDK.
All models are fully typed and validated for type safety and API contract compliance.

Example:
    >>> from gl_normalizer.models import NormalizeRequest, NormalizeResult
    >>> request = NormalizeRequest(value=100, unit="kWh", expected_dimension="energy")
    >>> print(request.model_dump_json())
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PolicyMode(str, Enum):
    """
    Policy mode for normalization behavior.

    Attributes:
        STRICT: Fail fast on any ambiguity or error.
        LENIENT: Emit warnings and continue processing.
    """

    STRICT = "STRICT"
    LENIENT = "LENIENT"


class BatchMode(str, Enum):
    """
    Batch processing mode for handling partial failures.

    Attributes:
        PARTIAL: Return results for successful items, errors for failed.
        ALL_OR_NOTHING: Fail entire batch if any item fails.
    """

    PARTIAL = "PARTIAL"
    ALL_OR_NOTHING = "ALL_OR_NOTHING"


class MatchMethod(str, Enum):
    """
    Method used for entity resolution matching.

    Attributes:
        EXACT_ID: Matched by exact reference ID.
        EXACT_NAME: Matched by exact canonical name.
        ALIAS: Matched by registered alias.
        RULE: Matched by rule-based normalization.
        FUZZY: Matched by fuzzy string matching.
        LLM_CANDIDATE: Suggested by LLM (requires review).
    """

    EXACT_ID = "exact_id"
    EXACT_NAME = "exact_name"
    ALIAS = "alias"
    RULE = "rule"
    FUZZY = "fuzzy"
    LLM_CANDIDATE = "llm_candidate"


class EntityType(str, Enum):
    """
    Type of entity for resolution.

    Attributes:
        FUEL: Fuel type entity.
        MATERIAL: Material entity.
        PROCESS: Process entity.
    """

    FUEL = "fuel"
    MATERIAL = "material"
    PROCESS = "process"


class JobStatus(str, Enum):
    """
    Status of an async normalization job.

    Attributes:
        PENDING: Job created but not started.
        PROCESSING: Job is currently processing.
        COMPLETED: Job completed successfully.
        FAILED: Job failed with errors.
        CANCELLED: Job was cancelled.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClientConfig(BaseModel):
    """
    Configuration options for NormalizerClient.

    Attributes:
        timeout: Request timeout in seconds (default: 30.0).
        max_retries: Maximum retry attempts for transient failures (default: 3).
        retry_delay: Initial delay between retries in seconds (default: 1.0).
        retry_max_delay: Maximum delay between retries in seconds (default: 30.0).
        enable_cache: Enable response caching (default: True).
        cache_ttl: Cache time-to-live in seconds (default: 300).
        pool_connections: Number of connection pool connections (default: 10).
        pool_maxsize: Maximum pool size (default: 100).

    Example:
        >>> config = ClientConfig(timeout=60.0, max_retries=5)
        >>> client = NormalizerClient(api_key="...", config=config)
    """

    model_config = ConfigDict(frozen=True)

    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, gt=0, description="Initial retry delay in seconds")
    retry_max_delay: float = Field(default=30.0, gt=0, description="Maximum retry delay in seconds")
    enable_cache: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=300, gt=0, description="Cache TTL in seconds")
    pool_connections: int = Field(default=10, gt=0, description="Connection pool size")
    pool_maxsize: int = Field(default=100, gt=0, description="Maximum pool size")


class ReferenceConditions(BaseModel):
    """
    Reference conditions for basis-dependent conversions.

    Required for volume units like Nm3 (normal cubic meters) and scf (standard cubic feet).

    Attributes:
        temperature_c: Reference temperature in Celsius.
        pressure_kpa: Reference pressure in kilopascals.

    Example:
        >>> conditions = ReferenceConditions(temperature_c=0, pressure_kpa=101.325)
    """

    model_config = ConfigDict(frozen=True)

    temperature_c: float = Field(
        default=0.0,
        alias="temperature_C",
        description="Reference temperature in Celsius",
    )
    pressure_kpa: float = Field(
        default=101.325,
        alias="pressure_kPa",
        description="Reference pressure in kilopascals",
    )


class NormalizeMetadata(BaseModel):
    """
    Optional metadata for normalization requests.

    Attributes:
        locale: Locale for parsing (e.g., "en-US", "de-DE").
        reference_conditions: Reference conditions for basis-dependent units.
        gwp_version: GWP version for CO2e conversions (AR5 or AR6).
        notes: Free-form notes for audit trail.

    Example:
        >>> metadata = NormalizeMetadata(gwp_version="AR6", locale="en-US")
    """

    model_config = ConfigDict(populate_by_name=True)

    locale: Optional[str] = Field(default=None, description="Locale for parsing")
    reference_conditions: Optional[ReferenceConditions] = Field(
        default=None, description="Reference conditions for volume conversions"
    )
    gwp_version: Optional[str] = Field(
        default=None, pattern=r"^AR[56]$", description="GWP version (AR5 or AR6)"
    )
    notes: Optional[str] = Field(default=None, max_length=1000, description="Free-form notes")


class NormalizeRequest(BaseModel):
    """
    Request model for normalizing a single measurement.

    Attributes:
        value: Numeric value to normalize.
        unit: Unit string (may be messy).
        target_unit: Optional target unit for conversion.
        expected_dimension: Expected dimension (e.g., "energy", "mass").
        field: Field name for audit trail.
        metadata: Optional metadata for the request.

    Example:
        >>> request = NormalizeRequest(
        ...     value=100,
        ...     unit="kWh",
        ...     target_unit="MJ",
        ...     expected_dimension="energy",
        ...     field="energy_consumption"
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    value: float = Field(..., description="Numeric value to normalize")
    unit: str = Field(..., min_length=1, max_length=500, description="Unit string")
    target_unit: Optional[str] = Field(
        default=None, max_length=500, description="Target unit for conversion"
    )
    expected_dimension: Optional[str] = Field(
        default=None, description="Expected dimension for validation"
    )
    field: Optional[str] = Field(
        default=None, max_length=255, description="Field name for audit trail"
    )
    metadata: Optional[NormalizeMetadata] = Field(default=None, description="Optional metadata")

    @field_validator("unit", "target_unit", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Strip leading/trailing whitespace from unit strings."""
        if v is not None:
            return v.strip()
        return v


class EntityHints(BaseModel):
    """
    Hints for entity resolution disambiguation.

    Attributes:
        region: Geographic region hint (e.g., "EU", "NA").
        sector: Industry sector hint (e.g., "energy", "manufacturing").
        taxonomy: Taxonomy hint (e.g., "GHG Protocol", "NAICS").

    Example:
        >>> hints = EntityHints(region="EU", sector="energy")
    """

    model_config = ConfigDict(frozen=True)

    region: Optional[str] = Field(default=None, max_length=50, description="Geographic region")
    sector: Optional[str] = Field(default=None, max_length=100, description="Industry sector")
    taxonomy: Optional[str] = Field(default=None, max_length=100, description="Taxonomy reference")


class EntityRequest(BaseModel):
    """
    Request model for entity resolution.

    Attributes:
        entity_type: Type of entity (fuel, material, process).
        raw_name: Raw name to resolve.
        raw_code: Optional raw code/identifier.
        field: Field name for audit trail.
        hints: Optional hints for disambiguation.

    Example:
        >>> request = EntityRequest(
        ...     entity_type=EntityType.FUEL,
        ...     raw_name="Nat Gas",
        ...     hints=EntityHints(region="EU")
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    entity_type: EntityType = Field(..., description="Type of entity to resolve")
    raw_name: str = Field(..., min_length=1, max_length=1000, description="Raw name to resolve")
    raw_code: Optional[str] = Field(default=None, max_length=255, description="Raw code/identifier")
    field: Optional[str] = Field(default=None, max_length=255, description="Field name")
    hints: Optional[EntityHints] = Field(default=None, description="Disambiguation hints")


class ConversionStep(BaseModel):
    """
    A single step in the conversion trace.

    Attributes:
        from_unit: Source unit.
        to_unit: Target unit.
        factor: Conversion factor applied.
        method: Conversion method (multiply, divide, affine).
        reference_conditions: Reference conditions if applicable.

    Example:
        >>> step = ConversionStep(from_unit="kWh", to_unit="MJ", factor=3.6, method="multiply")
    """

    model_config = ConfigDict(frozen=True)

    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")
    factor: float = Field(..., description="Conversion factor")
    method: str = Field(..., description="Conversion method")
    reference_conditions: Optional[Dict[str, float]] = Field(
        default=None, description="Reference conditions if applicable"
    )


class ConversionTrace(BaseModel):
    """
    Complete trace of conversion steps for audit.

    Attributes:
        steps: List of conversion steps.
        factor_version: Version of conversion factors used.

    Example:
        >>> trace = ConversionTrace(
        ...     steps=[ConversionStep(from_unit="kWh", to_unit="MJ", factor=3.6, method="multiply")],
        ...     factor_version="2026.01.0"
        ... )
    """

    model_config = ConfigDict(frozen=True)

    steps: List[ConversionStep] = Field(..., description="Conversion steps")
    factor_version: str = Field(..., description="Factor version used")


class Warning(BaseModel):
    """
    Warning message from normalization.

    Attributes:
        code: GLNORM warning code.
        severity: Warning severity (always "warning").
        path: JSON path to the affected field.
        message: Human-readable warning message.

    Example:
        >>> warning = Warning(
        ...     code="GLNORM-E307",
        ...     severity="warning",
        ...     path="/measurements/0",
        ...     message="Using deprecated conversion factor"
        ... )
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., description="GLNORM warning code")
    severity: str = Field(default="warning", description="Warning severity")
    path: Optional[str] = Field(default=None, description="JSON path")
    message: str = Field(..., description="Warning message")


class NormalizeResult(BaseModel):
    """
    Result of normalizing a single measurement.

    Attributes:
        field: Field name from request.
        dimension: Computed dimension (e.g., "energy").
        canonical_value: Normalized value in canonical units.
        canonical_unit: Canonical unit string.
        raw_value: Original input value.
        raw_unit: Original input unit.
        conversion_trace: Detailed conversion steps for audit.
        warnings: List of warnings generated.

    Example:
        >>> result = client.normalize(100, "kWh", target_unit="MJ")
        >>> print(result.canonical_value)  # 360.0
        >>> print(result.canonical_unit)   # "MJ"
    """

    model_config = ConfigDict(frozen=True)

    field: Optional[str] = Field(default=None, description="Field name")
    dimension: str = Field(..., description="Computed dimension")
    canonical_value: float = Field(..., description="Normalized value")
    canonical_unit: str = Field(..., description="Canonical unit")
    raw_value: float = Field(..., description="Original value")
    raw_unit: str = Field(..., description="Original unit")
    conversion_trace: Optional[ConversionTrace] = Field(
        default=None, description="Conversion audit trail"
    )
    warnings: List[Warning] = Field(default_factory=list, description="Warnings")


class EntityResult(BaseModel):
    """
    Result of resolving an entity.

    Attributes:
        field: Field name from request.
        entity_type: Type of entity.
        raw_name: Original input name.
        reference_id: Resolved reference ID.
        canonical_name: Canonical entity name.
        vocabulary_version: Vocabulary version used.
        match_method: Method used for matching.
        confidence: Confidence score (0.0-1.0).
        needs_review: Whether human review is recommended.
        warnings: List of warnings generated.

    Example:
        >>> result = client.resolve_entity("Nat Gas", entity_type="fuel")
        >>> print(result.reference_id)      # "GL-FUEL-NATGAS"
        >>> print(result.canonical_name)    # "Natural gas"
        >>> print(result.confidence)        # 1.0
    """

    model_config = ConfigDict(frozen=True)

    field: Optional[str] = Field(default=None, description="Field name")
    entity_type: EntityType = Field(..., description="Entity type")
    raw_name: str = Field(..., description="Original name")
    reference_id: str = Field(..., description="Resolved reference ID")
    canonical_name: str = Field(..., description="Canonical name")
    vocabulary_version: str = Field(..., description="Vocabulary version")
    match_method: MatchMethod = Field(..., description="Match method used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    needs_review: bool = Field(default=False, description="Needs human review")
    warnings: List[Warning] = Field(default_factory=list, description="Warnings")


class AuditInfo(BaseModel):
    """
    Audit information from normalization.

    Attributes:
        normalization_event_id: Unique ID for the normalization event.
        status: Status of the normalization (success, warning, failed).

    Example:
        >>> print(result.audit.normalization_event_id)
        "norm-evt-abc123"
    """

    model_config = ConfigDict(frozen=True)

    normalization_event_id: str = Field(..., description="Normalization event ID")
    status: str = Field(..., description="Normalization status")


class BatchSummary(BaseModel):
    """
    Summary statistics for batch normalization.

    Attributes:
        total: Total number of items in batch.
        success: Number of successful normalizations.
        failed: Number of failed normalizations.
        warnings: Number of items with warnings.

    Example:
        >>> print(f"Success rate: {result.summary.success / result.summary.total * 100}%")
    """

    model_config = ConfigDict(frozen=True)

    total: int = Field(..., ge=0, description="Total items")
    success: int = Field(..., ge=0, description="Successful items")
    failed: int = Field(..., ge=0, description="Failed items")
    warnings: int = Field(default=0, ge=0, description="Items with warnings")


class BatchItemResult(BaseModel):
    """
    Result for a single item in a batch.

    Attributes:
        source_record_id: Source record identifier.
        status: Item status (success, warning, failed).
        canonical_measurements: Normalized measurements.
        normalized_entities: Resolved entities.
        audit: Audit information.
        errors: List of errors if failed.
        warnings: List of warnings.

    Example:
        >>> for item in result.results:
        ...     if item.status == "success":
        ...         print(item.canonical_measurements[0].canonical_value)
    """

    model_config = ConfigDict(frozen=True)

    source_record_id: Optional[str] = Field(default=None, description="Source record ID")
    status: str = Field(..., description="Item status")
    canonical_measurements: List[NormalizeResult] = Field(
        default_factory=list, description="Normalized measurements"
    )
    normalized_entities: List[EntityResult] = Field(
        default_factory=list, description="Resolved entities"
    )
    audit: Optional[AuditInfo] = Field(default=None, description="Audit info")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Errors")
    warnings: List[Warning] = Field(default_factory=list, description="Warnings")


class BatchResult(BaseModel):
    """
    Result of batch normalization.

    Attributes:
        summary: Summary statistics.
        results: Per-item results.

    Example:
        >>> result = client.normalize_batch(requests)
        >>> print(f"Processed {result.summary.total} items")
        >>> print(f"Success: {result.summary.success}, Failed: {result.summary.failed}")
    """

    model_config = ConfigDict(frozen=True)

    summary: BatchSummary = Field(..., description="Batch summary")
    results: List[BatchItemResult] = Field(..., description="Per-item results")


class Job(BaseModel):
    """
    Async normalization job.

    Attributes:
        job_id: Unique job identifier.
        status: Current job status.
        progress: Progress percentage (0-100).
        total_items: Total items to process.
        processed_items: Items processed so far.
        created_at: Job creation timestamp.
        updated_at: Last update timestamp.
        completed_at: Completion timestamp (if completed).
        results_url: URL to download results (if completed).
        error: Error message (if failed).

    Example:
        >>> job = client.create_job(requests)
        >>> while job.status == JobStatus.PROCESSING:
        ...     time.sleep(5)
        ...     job = client.get_job(job.job_id)
        ...     print(f"Progress: {job.progress}%")
    """

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    total_items: int = Field(default=0, ge=0, description="Total items")
    processed_items: int = Field(default=0, ge=0, description="Processed items")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    results_url: Optional[str] = Field(default=None, description="Results download URL")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class Vocabulary(BaseModel):
    """
    Vocabulary metadata.

    Attributes:
        vocabulary_id: Unique vocabulary identifier.
        name: Human-readable name.
        version: Semantic version string.
        entity_type: Type of entities in vocabulary.
        entity_count: Number of entities.
        created_at: Creation timestamp.
        description: Optional description.

    Example:
        >>> vocabs = client.list_vocabularies()
        >>> for vocab in vocabs:
        ...     print(f"{vocab.name} v{vocab.version}: {vocab.entity_count} entities")
    """

    model_config = ConfigDict(frozen=True)

    vocabulary_id: str = Field(..., description="Vocabulary identifier")
    name: str = Field(..., description="Vocabulary name")
    version: str = Field(..., description="Semantic version")
    entity_type: EntityType = Field(..., description="Entity type")
    entity_count: int = Field(..., ge=0, description="Number of entities")
    created_at: datetime = Field(..., description="Creation timestamp")
    description: Optional[str] = Field(default=None, description="Description")


class VocabularyEntry(BaseModel):
    """
    Single entry in a vocabulary.

    Attributes:
        reference_id: Stable reference identifier.
        canonical_name: Canonical entity name.
        entity_type: Type of entity.
        status: Entry status (active, deprecated).
        replaced_by: Reference ID of replacement (if deprecated).
        aliases: List of known aliases.
        attributes: Additional attributes.

    Example:
        >>> entry = client.get_vocabulary_entry("GL-FUEL-NATGAS")
        >>> print(entry.canonical_name)  # "Natural gas"
        >>> print(entry.aliases)         # ["Nat Gas", "Natural-gas", ...]
    """

    model_config = ConfigDict(frozen=True)

    reference_id: str = Field(..., description="Reference identifier")
    canonical_name: str = Field(..., description="Canonical name")
    entity_type: EntityType = Field(..., description="Entity type")
    status: str = Field(default="active", description="Entry status")
    replaced_by: Optional[str] = Field(default=None, description="Replacement reference ID")
    aliases: List[str] = Field(default_factory=list, description="Known aliases")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")
