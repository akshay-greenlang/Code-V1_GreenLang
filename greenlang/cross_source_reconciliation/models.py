# -*- coding: utf-8 -*-
"""
Cross-Source Reconciliation Agent Service Data Models - AGENT-DATA-015

Pydantic v2 data models for the Cross-Source Reconciliation SDK. Defines
enumerations, SDK models, request models, and constants for multi-source
data matching, field-level comparison, discrepancy detection, conflict
resolution, golden record assembly, and reconciliation pipeline
orchestration.

Re-exported Layer 1 sources:
    - greenlang.data_quality_profiler.consistency_analyzer:
        ConsistencyAnalyzer (as L1ConsistencyAnalyzer)
    - greenlang.duplicate_detector.similarity_scorer:
        SimilarityScorer (as L1SimilarityScorer)
    - greenlang.duplicate_detector.match_classifier:
        MatchClassifier (as L1MatchClassifier)
    - greenlang.data.data_engineering.reconciliation.factor_reconciliation:
        FactorReconciler (as L1FactorReconciler),
        ConflictResolutionStrategy (as L1ConflictResolutionStrategy)

New enumerations (13):
    - SourceType, SourceStatus, MatchStrategy, MatchStatus,
      ComparisonResult, DiscrepancyType, DiscrepancySeverity,
      ResolutionStrategy, ResolutionStatus, FieldType,
      ReconciliationStatus, TemporalGranularity, CredibilityFactor

New SDK models (22):
    - SourceDefinition, SchemaMapping, MatchKey, MatchResult,
      FieldComparison, Discrepancy, ResolutionDecision, GoldenRecord,
      SourceCredibility, ToleranceRule, ReconciliationReport,
      ReconciliationJobConfig, BatchMatchResult, DiscrepancySummary,
      ResolutionSummary, TemporalAlignment, FieldLineage,
      ReconciliationStats, SourceHealthMetrics, ComparisonSummary,
      PipelineStageResult, ReconciliationEvent

Request models (8):
    - CreateJobRequest, RegisterSourceRequest, UpdateSourceRequest,
      MatchRequest, CompareRequest, ResolveRequest,
      PipelineRequest, GoldenRecordRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from data_quality_profiler
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.consistency_analyzer import (
        ConsistencyAnalyzer as L1ConsistencyAnalyzer,
    )
    ConsistencyAnalyzer = L1ConsistencyAnalyzer
except ImportError:
    ConsistencyAnalyzer = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from duplicate_detector
# ---------------------------------------------------------------------------

try:
    from greenlang.duplicate_detector.similarity_scorer import (
        SimilarityScorer as L1SimilarityScorer,
    )
    SimilarityScorer = L1SimilarityScorer
except ImportError:
    SimilarityScorer = None  # type: ignore[assignment, misc]

try:
    from greenlang.duplicate_detector.match_classifier import (
        MatchClassifier as L1MatchClassifier,
    )
    MatchClassifier = L1MatchClassifier
except ImportError:
    MatchClassifier = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from data_engineering reconciliation
# ---------------------------------------------------------------------------

try:
    from greenlang.data.data_engineering.reconciliation.factor_reconciliation import (
        FactorReconciler as L1FactorReconciler,
    )
    FactorReconciler = L1FactorReconciler
except ImportError:
    FactorReconciler = None  # type: ignore[assignment, misc]

try:
    from greenlang.data.data_engineering.reconciliation.factor_reconciliation import (
        ConflictResolutionStrategy as L1ConflictResolutionStrategy,
    )
    ConflictResolutionStrategy = L1ConflictResolutionStrategy
except ImportError:
    ConflictResolutionStrategy = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default matching confidence threshold for accepting a match pair.
DEFAULT_MATCH_THRESHOLD: float = 0.85

#: Default relative tolerance percentage for numeric field comparisons.
DEFAULT_TOLERANCE_PCT: float = 5.0

#: Default absolute tolerance for numeric field comparisons.
DEFAULT_TOLERANCE_ABS: float = 0.01

#: Maximum number of data sources that can participate in a single job.
MAX_SOURCES: int = 20

#: Maximum number of candidate match pairs returned per entity.
MAX_MATCH_CANDIDATES: int = 100

#: Deviation percentage threshold that qualifies as CRITICAL severity.
CRITICAL_THRESHOLD_PCT: float = 50.0

#: Deviation percentage threshold that qualifies as HIGH severity.
HIGH_THRESHOLD_PCT: float = 25.0

#: Deviation percentage threshold that qualifies as MEDIUM severity.
MEDIUM_THRESHOLD_PCT: float = 10.0

#: Supported unit conversions for automatic cross-source normalization.
SUPPORTED_UNITS: Dict[str, float] = {
    "kg_to_tonnes": 0.001,
    "tonnes_to_kg": 1000.0,
    "g_to_kg": 0.001,
    "kg_to_g": 1000.0,
    "lb_to_kg": 0.453592,
    "kg_to_lb": 2.20462,
    "MWh_to_kWh": 1000.0,
    "kWh_to_MWh": 0.001,
    "GJ_to_MWh": 0.277778,
    "MWh_to_GJ": 3.6,
    "m3_to_litre": 1000.0,
    "litre_to_m3": 0.001,
    "gallon_to_litre": 3.78541,
    "litre_to_gallon": 0.264172,
    "mile_to_km": 1.60934,
    "km_to_mile": 0.621371,
    "tCO2e_to_kgCO2e": 1000.0,
    "kgCO2e_to_tCO2e": 0.001,
}

#: Supported ISO 4217 currency codes for multi-currency reconciliation.
SUPPORTED_CURRENCIES: List[str] = [
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD", "CHF", "SEK",
    "NOK", "DKK", "NZD", "SGD", "HKD", "KRW", "BRL", "ZAR", "MXN", "AED",
    "SAR", "THB", "MYR", "IDR", "PHP", "TWD", "PLN", "CZK", "HUF", "TRY",
]


# =============================================================================
# Enumerations (13)
# =============================================================================


class SourceType(str, Enum):
    """Classification of the external data source providing records.

    ERP: Enterprise resource planning system (SAP, Oracle, etc.).
    UTILITY: Utility provider data feed (electricity, gas, water).
    METER: Direct metering or IoT device readings.
    QUESTIONNAIRE: Supplier or stakeholder questionnaire responses.
    SPREADSHEET: Uploaded spreadsheet or CSV file.
    API: Third-party API integration.
    IOT: Internet-of-Things sensor platform.
    REGISTRY: Official registry or certification body.
    MANUAL: Manually entered data by an operator.
    OTHER: Any source not covered by the above categories.
    """

    ERP = "erp"
    UTILITY = "utility"
    METER = "meter"
    QUESTIONNAIRE = "questionnaire"
    SPREADSHEET = "spreadsheet"
    API = "api"
    IOT = "iot"
    REGISTRY = "registry"
    MANUAL = "manual"
    OTHER = "other"


class SourceStatus(str, Enum):
    """Operational status of a registered data source.

    ACTIVE: Source is live and contributing data.
    INACTIVE: Source is registered but not currently providing data.
    ERROR: Source encountered an error during last ingestion.
    PENDING_VALIDATION: Source is awaiting initial validation.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING_VALIDATION = "pending_validation"


class MatchStrategy(str, Enum):
    """Algorithm used to identify matching records across sources.

    EXACT: Exact key equality on all match fields.
    FUZZY: Similarity-based matching with configurable threshold.
    COMPOSITE: Multi-field weighted matching combining exact and fuzzy.
    TEMPORAL: Time-window based matching for period alignment.
    BLOCKING: Blocking-key partitioning followed by pairwise comparison.
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    COMPOSITE = "composite"
    TEMPORAL = "temporal"
    BLOCKING = "blocking"


class MatchStatus(str, Enum):
    """Outcome status of a record-matching attempt.

    MATCHED: Records were confidently matched across sources.
    UNMATCHED: No matching record found in the partner source.
    AMBIGUOUS: Multiple candidate matches with insufficient differentiation.
    PENDING_REVIEW: Match requires human review before confirmation.
    """

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"
    PENDING_REVIEW = "pending_review"


class ComparisonResult(str, Enum):
    """Outcome of comparing a single field across two source records.

    MATCH: Field values are identical (or equivalent after normalization).
    MISMATCH: Field values differ beyond the configured tolerance.
    WITHIN_TOLERANCE: Field values differ but are within tolerance.
    MISSING_LEFT: Field is missing in the left (source A) record.
    MISSING_RIGHT: Field is missing in the right (source B) record.
    INCOMPARABLE: Fields cannot be compared (incompatible types or units).
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    WITHIN_TOLERANCE = "within_tolerance"
    MISSING_LEFT = "missing_left"
    MISSING_RIGHT = "missing_right"
    INCOMPARABLE = "incomparable"


class DiscrepancyType(str, Enum):
    """Classification of a detected discrepancy between sources.

    VALUE_MISMATCH: Numeric or categorical values disagree.
    MISSING_IN_SOURCE: Record or field is absent from one source.
    EXTRA_IN_SOURCE: Record or field appears in one source but not the other.
    TIMING_DIFFERENCE: Records refer to different periods or timestamps.
    UNIT_DIFFERENCE: Values reported in different units without conversion.
    AGGREGATION_MISMATCH: Aggregated totals do not reconcile with line items.
    FORMAT_DIFFERENCE: Same value represented in different formats.
    """

    VALUE_MISMATCH = "value_mismatch"
    MISSING_IN_SOURCE = "missing_in_source"
    EXTRA_IN_SOURCE = "extra_in_source"
    TIMING_DIFFERENCE = "timing_difference"
    UNIT_DIFFERENCE = "unit_difference"
    AGGREGATION_MISMATCH = "aggregation_mismatch"
    FORMAT_DIFFERENCE = "format_difference"


class DiscrepancySeverity(str, Enum):
    """Severity level of a detected discrepancy.

    CRITICAL: Deviation exceeds 50 pct; requires immediate action.
    HIGH: Deviation between 25-50 pct; escalation recommended.
    MEDIUM: Deviation between 10-25 pct; should be reviewed.
    LOW: Deviation below 10 pct; informational only.
    INFO: No material impact; logged for audit completeness.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ResolutionStrategy(str, Enum):
    """Strategy for resolving a detected discrepancy.

    PRIORITY_WINS: The source with the highest priority wins.
    MOST_RECENT: The most recently updated value wins.
    WEIGHTED_AVERAGE: Compute a credibility-weighted average.
    MOST_COMPLETE: The source with the fewest missing fields wins.
    CONSENSUS: Majority-vote across 3+ sources wins.
    MANUAL_REVIEW: Route the discrepancy for human review.
    CUSTOM: Apply a user-defined resolution function.
    """

    PRIORITY_WINS = "priority_wins"
    MOST_RECENT = "most_recent"
    WEIGHTED_AVERAGE = "weighted_average"
    MOST_COMPLETE = "most_complete"
    CONSENSUS = "consensus"
    MANUAL_REVIEW = "manual_review"
    CUSTOM = "custom"


class ResolutionStatus(str, Enum):
    """Lifecycle status of a discrepancy resolution decision.

    RESOLVED: Discrepancy has been successfully resolved.
    PENDING_REVIEW: Resolution is awaiting human review.
    ESCALATED: Resolution has been escalated to a senior reviewer.
    DEFERRED: Resolution has been deferred to a later cycle.
    REJECTED: Proposed resolution was rejected.
    """

    RESOLVED = "resolved"
    PENDING_REVIEW = "pending_review"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class FieldType(str, Enum):
    """Data type classification for a compared field.

    NUMERIC: Floating-point or integer values.
    STRING: Free-text or coded string values.
    DATE: Date or datetime values.
    BOOLEAN: True/false values.
    CATEGORICAL: Enumerated category values from a controlled vocabulary.
    CURRENCY: Monetary values with an associated currency code.
    UNIT_VALUE: Numeric values with an associated unit of measurement.
    """

    NUMERIC = "numeric"
    STRING = "string"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    CURRENCY = "currency"
    UNIT_VALUE = "unit_value"


class ReconciliationStatus(str, Enum):
    """Lifecycle status of a reconciliation job.

    PENDING: Job has been created but not yet started.
    RUNNING: Job is currently executing.
    COMPLETED: Job finished successfully.
    FAILED: Job terminated due to an error.
    CANCELLED: Job was cancelled by the user.
    PARTIAL: Job completed but some records could not be reconciled.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class TemporalGranularity(str, Enum):
    """Time granularity for temporal alignment of source records.

    HOURLY: Align records to the nearest hour.
    DAILY: Align records to the nearest day.
    WEEKLY: Align records to the nearest ISO week.
    MONTHLY: Align records to the nearest calendar month.
    QUARTERLY: Align records to the nearest calendar quarter.
    ANNUAL: Align records to the nearest calendar year.
    """

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class CredibilityFactor(str, Enum):
    """Dimension contributing to a source's overall credibility score.

    COMPLETENESS: Fraction of expected fields that are present.
    TIMELINESS: How recently the source data was refreshed.
    CONSISTENCY: Internal consistency of the source's own records.
    ACCURACY: Historical accuracy compared to verified benchmarks.
    CERTIFICATION: Presence of third-party certification or audit.
    """

    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    CERTIFICATION = "certification"


# =============================================================================
# SDK Data Models (22)
# =============================================================================


class SourceDefinition(BaseModel):
    """A registered data source participating in cross-source reconciliation.

    Contains metadata, priority, credibility scoring, and operational
    status for a single external data source.

    Attributes:
        id: Unique identifier for this source.
        name: Human-readable source name.
        source_type: Classification of the source system.
        priority: Source priority for conflict resolution (1=lowest, 100=highest).
        credibility_score: Overall credibility score (0.0-1.0).
        schema_info: Schema metadata describing the source's field structure.
        refresh_cadence: How frequently the source refreshes (e.g. 'daily', 'weekly').
        description: Human-readable description of the source.
        tags: List of tags for categorization and filtering.
        status: Current operational status of the source.
        created_at: When the source was registered.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this source",
    )
    name: str = Field(
        ..., description="Human-readable source name",
    )
    source_type: SourceType = Field(
        default=SourceType.OTHER,
        description="Classification of the source system",
    )
    priority: int = Field(
        default=50, ge=1, le=100,
        description="Source priority for conflict resolution (1=lowest, 100=highest)",
    )
    credibility_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Overall credibility score (0.0-1.0)",
    )
    schema_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema metadata describing the source's field structure",
    )
    refresh_cadence: str = Field(
        default="daily",
        description="How frequently the source refreshes (e.g. 'daily', 'weekly')",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the source",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorization and filtering",
    )
    status: SourceStatus = Field(
        default=SourceStatus.PENDING_VALIDATION,
        description="Current operational status of the source",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the source was registered",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class SchemaMapping(BaseModel):
    """Column-level mapping from a source schema to the canonical schema.

    Defines how a single column in a source dataset maps to the
    canonical reconciliation schema, including optional transforms
    and unit conversions.

    Attributes:
        source_column: Column name in the source dataset.
        canonical_column: Corresponding column name in the canonical schema.
        transform: Optional transformation expression to apply.
        unit_from: Unit of measurement in the source (if applicable).
        unit_to: Target canonical unit of measurement (if applicable).
        date_format: Date format string in the source (e.g. '%Y-%m-%d').
    """

    source_column: str = Field(
        ..., description="Column name in the source dataset",
    )
    canonical_column: str = Field(
        ..., description="Corresponding column name in the canonical schema",
    )
    transform: Optional[str] = Field(
        None,
        description="Optional transformation expression to apply",
    )
    unit_from: Optional[str] = Field(
        None,
        description="Unit of measurement in the source (if applicable)",
    )
    unit_to: Optional[str] = Field(
        None,
        description="Target canonical unit of measurement (if applicable)",
    )
    date_format: Optional[str] = Field(
        None,
        description="Date format string in the source (e.g. '%Y-%m-%d')",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_column")
    @classmethod
    def validate_source_column(cls, v: str) -> str:
        """Validate source_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_column must be non-empty")
        return v

    @field_validator("canonical_column")
    @classmethod
    def validate_canonical_column(cls, v: str) -> str:
        """Validate canonical_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("canonical_column must be non-empty")
        return v


class MatchKey(BaseModel):
    """Composite key used to identify a record for cross-source matching.

    Combines entity, period, metric, and source identifiers into a
    single hashable key for efficient pair-wise comparison.

    Attributes:
        entity_id: Identifier of the entity (facility, supplier, etc.).
        period: Reporting period string (e.g. '2025-Q1', '2025-01').
        metric_name: Name of the metric being reconciled.
        source_id: Identifier of the data source.
        composite_key: Computed composite key combining all fields.
    """

    entity_id: str = Field(
        ..., description="Identifier of the entity (facility, supplier, etc.)",
    )
    period: str = Field(
        ..., description="Reporting period string (e.g. '2025-Q1', '2025-01')",
    )
    metric_name: str = Field(
        ..., description="Name of the metric being reconciled",
    )
    source_id: str = Field(
        ..., description="Identifier of the data source",
    )
    composite_key: str = Field(
        default="",
        description="Computed composite key combining all fields",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @model_validator(mode="after")
    def compute_composite_key(self) -> MatchKey:
        """Compute composite_key from entity_id, period, metric_name, source_id."""
        if not self.composite_key:
            raw = f"{self.entity_id}|{self.period}|{self.metric_name}|{self.source_id}"
            self.composite_key = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self


class MatchResult(BaseModel):
    """Result of matching a pair of records across two sources.

    Contains the match identifiers, confidence score, strategy used,
    matched fields, and provenance hash for full auditability.

    Attributes:
        match_id: Unique identifier for this match result.
        source_a_key: Match key from source A.
        source_b_key: Match key from source B.
        confidence: Match confidence score (0.0-1.0).
        strategy: Matching strategy that produced this result.
        status: Outcome status of the match attempt.
        matched_fields: List of field names that contributed to the match.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    match_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this match result",
    )
    source_a_key: MatchKey = Field(
        ..., description="Match key from source A",
    )
    source_b_key: MatchKey = Field(
        ..., description="Match key from source B",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Match confidence score (0.0-1.0)",
    )
    strategy: MatchStrategy = Field(
        default=MatchStrategy.EXACT,
        description="Matching strategy that produced this result",
    )
    status: MatchStatus = Field(
        default=MatchStatus.PENDING_REVIEW,
        description="Outcome status of the match attempt",
    )
    matched_fields: List[str] = Field(
        default_factory=list,
        description="List of field names that contributed to the match",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class FieldComparison(BaseModel):
    """Result of comparing a single field across two matched records.

    Reports absolute and relative differences, tolerance checks,
    and the overall comparison outcome for one field.

    Attributes:
        field_name: Name of the compared field.
        field_type: Data type classification of the field.
        source_a_value: Value from source A.
        source_b_value: Value from source B.
        absolute_diff: Absolute difference between values (numeric fields).
        relative_diff_pct: Relative difference as a percentage (numeric fields).
        tolerance_abs: Absolute tolerance applied to this comparison.
        tolerance_pct: Percentage tolerance applied to this comparison.
        result: Outcome of the field comparison.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    field_name: str = Field(
        ..., description="Name of the compared field",
    )
    field_type: FieldType = Field(
        default=FieldType.NUMERIC,
        description="Data type classification of the field",
    )
    source_a_value: Any = Field(
        default=None,
        description="Value from source A",
    )
    source_b_value: Any = Field(
        default=None,
        description="Value from source B",
    )
    absolute_diff: Optional[float] = Field(
        None,
        description="Absolute difference between values (numeric fields)",
    )
    relative_diff_pct: Optional[float] = Field(
        None,
        description="Relative difference as a percentage (numeric fields)",
    )
    tolerance_abs: Optional[float] = Field(
        None,
        description="Absolute tolerance applied to this comparison",
    )
    tolerance_pct: Optional[float] = Field(
        None,
        description="Percentage tolerance applied to this comparison",
    )
    result: ComparisonResult = Field(
        default=ComparisonResult.INCOMPARABLE,
        description="Outcome of the field comparison",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class Discrepancy(BaseModel):
    """A detected discrepancy between two matched source records.

    Records the type, severity, affected values, and deviation
    for a single field-level disagreement between sources.

    Attributes:
        discrepancy_id: Unique identifier for this discrepancy.
        match_id: Identifier of the parent match result.
        field_name: Name of the field where the discrepancy was detected.
        discrepancy_type: Classification of the discrepancy.
        severity: Severity level of the discrepancy.
        source_a_value: Value from source A.
        source_b_value: Value from source B.
        deviation_pct: Percentage deviation between values.
        description: Human-readable description of the discrepancy.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    discrepancy_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this discrepancy",
    )
    match_id: str = Field(
        ..., description="Identifier of the parent match result",
    )
    field_name: str = Field(
        ..., description="Name of the field where the discrepancy was detected",
    )
    discrepancy_type: DiscrepancyType = Field(
        default=DiscrepancyType.VALUE_MISMATCH,
        description="Classification of the discrepancy",
    )
    severity: DiscrepancySeverity = Field(
        default=DiscrepancySeverity.LOW,
        description="Severity level of the discrepancy",
    )
    source_a_value: Any = Field(
        default=None,
        description="Value from source A",
    )
    source_b_value: Any = Field(
        default=None,
        description="Value from source B",
    )
    deviation_pct: Optional[float] = Field(
        None,
        description="Percentage deviation between values",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the discrepancy",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("match_id")
    @classmethod
    def validate_match_id(cls, v: str) -> str:
        """Validate match_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("match_id must be non-empty")
        return v

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class ResolutionDecision(BaseModel):
    """A resolution decision for a detected discrepancy.

    Records which strategy was applied, the winning source, resolved
    value, confidence, and optional human reviewer provenance.

    Attributes:
        resolution_id: Unique identifier for this resolution decision.
        discrepancy_id: Identifier of the resolved discrepancy.
        strategy: Resolution strategy applied.
        winning_source_id: Identifier of the source whose value was selected.
        resolved_value: The final resolved value.
        confidence: Confidence in the resolution decision (0.0-1.0).
        justification: Explanation of why this resolution was chosen.
        reviewer: Identifier of the human reviewer (if manual review).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    resolution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this resolution decision",
    )
    discrepancy_id: str = Field(
        ..., description="Identifier of the resolved discrepancy",
    )
    strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.PRIORITY_WINS,
        description="Resolution strategy applied",
    )
    winning_source_id: str = Field(
        default="",
        description="Identifier of the source whose value was selected",
    )
    resolved_value: Any = Field(
        default=None,
        description="The final resolved value",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the resolution decision (0.0-1.0)",
    )
    justification: str = Field(
        default="",
        description="Explanation of why this resolution was chosen",
    )
    reviewer: Optional[str] = Field(
        None,
        description="Identifier of the human reviewer (if manual review)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("discrepancy_id")
    @classmethod
    def validate_discrepancy_id(cls, v: str) -> str:
        """Validate discrepancy_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("discrepancy_id must be non-empty")
        return v


class GoldenRecord(BaseModel):
    """A single authoritative golden record assembled from multiple sources.

    Contains the best-available value for every field, along with
    the contributing source and per-field confidence scores.

    Attributes:
        record_id: Unique identifier for this golden record.
        entity_id: Identifier of the reconciled entity.
        period: Reporting period for this golden record.
        fields: Mapping of field names to their resolved values.
        field_sources: Mapping of field names to the source that provided the value.
        field_confidences: Mapping of field names to confidence scores (0.0-1.0).
        total_confidence: Aggregate confidence across all fields (0.0-1.0).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this golden record",
    )
    entity_id: str = Field(
        ..., description="Identifier of the reconciled entity",
    )
    period: str = Field(
        ..., description="Reporting period for this golden record",
    )
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of field names to their resolved values",
    )
    field_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of field names to the source that provided the value",
    )
    field_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of field names to confidence scores (0.0-1.0)",
    )
    total_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Aggregate confidence across all fields (0.0-1.0)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        """Validate period is non-empty."""
        if not v or not v.strip():
            raise ValueError("period must be non-empty")
        return v


class SourceCredibility(BaseModel):
    """Multi-dimensional credibility assessment for a data source.

    Scores the source across five dimensions (completeness, timeliness,
    consistency, accuracy, certification) and computes an overall score.

    Attributes:
        source_id: Identifier of the assessed source.
        completeness_score: Completeness dimension score (0.0-1.0).
        timeliness_score: Timeliness dimension score (0.0-1.0).
        consistency_score: Consistency dimension score (0.0-1.0).
        accuracy_score: Accuracy dimension score (0.0-1.0).
        certification_score: Certification dimension score (0.0-1.0).
        overall_score: Weighted overall credibility score (0.0-1.0).
        sample_size: Number of records used for assessment.
        last_assessed: Timestamp of the most recent assessment.
    """

    source_id: str = Field(
        ..., description="Identifier of the assessed source",
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Completeness dimension score (0.0-1.0)",
    )
    timeliness_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Timeliness dimension score (0.0-1.0)",
    )
    consistency_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Consistency dimension score (0.0-1.0)",
    )
    accuracy_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Accuracy dimension score (0.0-1.0)",
    )
    certification_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Certification dimension score (0.0-1.0)",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall credibility score (0.0-1.0)",
    )
    sample_size: int = Field(
        default=0, ge=0,
        description="Number of records used for assessment",
    )
    last_assessed: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the most recent assessment",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v


class ToleranceRule(BaseModel):
    """Tolerance configuration for a single field comparison.

    Defines absolute and percentage thresholds, rounding precision,
    unit conversion epsilon, and an optional custom comparator.

    Attributes:
        field_name: Name of the field this rule applies to.
        field_type: Data type of the field.
        tolerance_abs: Maximum absolute deviation allowed.
        tolerance_pct: Maximum relative deviation percentage allowed.
        rounding_digits: Number of decimal places for rounding before comparison.
        unit_conversion_epsilon: Epsilon for unit conversion rounding errors.
        custom_comparator: Optional name of a registered custom comparator function.
    """

    field_name: str = Field(
        ..., description="Name of the field this rule applies to",
    )
    field_type: FieldType = Field(
        default=FieldType.NUMERIC,
        description="Data type of the field",
    )
    tolerance_abs: Optional[float] = Field(
        None, ge=0.0,
        description="Maximum absolute deviation allowed",
    )
    tolerance_pct: Optional[float] = Field(
        None, ge=0.0,
        description="Maximum relative deviation percentage allowed",
    )
    rounding_digits: Optional[int] = Field(
        None, ge=0,
        description="Number of decimal places for rounding before comparison",
    )
    unit_conversion_epsilon: float = Field(
        default=1e-6, ge=0.0,
        description="Epsilon for unit conversion rounding errors",
    )
    custom_comparator: Optional[str] = Field(
        None,
        description="Optional name of a registered custom comparator function",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class ReconciliationReport(BaseModel):
    """Compliance-grade report summarizing a reconciliation job.

    Provides aggregate counts, resolution statistics, and a complete
    audit trail for regulatory submissions.

    Attributes:
        report_id: Unique identifier for this report.
        job_id: Identifier of the parent reconciliation job.
        total_records: Total number of records across all sources.
        matched_records: Number of successfully matched record pairs.
        discrepancies_found: Total number of discrepancies detected.
        discrepancies_resolved: Number of discrepancies successfully resolved.
        golden_records_created: Number of golden records assembled.
        unresolved_count: Number of discrepancies still unresolved.
        summary: Human-readable summary of the reconciliation results.
        created_at: When the report was generated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    job_id: str = Field(
        ..., description="Identifier of the parent reconciliation job",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of records across all sources",
    )
    matched_records: int = Field(
        default=0, ge=0,
        description="Number of successfully matched record pairs",
    )
    discrepancies_found: int = Field(
        default=0, ge=0,
        description="Total number of discrepancies detected",
    )
    discrepancies_resolved: int = Field(
        default=0, ge=0,
        description="Number of discrepancies successfully resolved",
    )
    golden_records_created: int = Field(
        default=0, ge=0,
        description="Number of golden records assembled",
    )
    unresolved_count: int = Field(
        default=0, ge=0,
        description="Number of discrepancies still unresolved",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of the reconciliation results",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the report was generated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("job_id must be non-empty")
        return v


class ReconciliationJobConfig(BaseModel):
    """Configuration for a reconciliation job run.

    Specifies which sources to reconcile, the matching and resolution
    strategies, tolerance rules, and feature toggles.

    Attributes:
        job_name: Human-readable name for the reconciliation job.
        source_ids: List of source identifiers to include.
        match_strategy: Record matching strategy to use.
        resolution_strategy: Discrepancy resolution strategy to use.
        tolerance_rules: List of field-level tolerance rules.
        enable_golden_records: Whether to assemble golden records.
        enable_temporal_alignment: Whether to perform temporal alignment.
        temporal_granularity: Temporal granularity for alignment (if enabled).
    """

    job_name: str = Field(
        default="",
        description="Human-readable name for the reconciliation job",
    )
    source_ids: List[str] = Field(
        ..., min_length=2,
        description="List of source identifiers to include",
    )
    match_strategy: MatchStrategy = Field(
        default=MatchStrategy.COMPOSITE,
        description="Record matching strategy to use",
    )
    resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.PRIORITY_WINS,
        description="Discrepancy resolution strategy to use",
    )
    tolerance_rules: List[ToleranceRule] = Field(
        default_factory=list,
        description="List of field-level tolerance rules",
    )
    enable_golden_records: bool = Field(
        default=True,
        description="Whether to assemble golden records",
    )
    enable_temporal_alignment: bool = Field(
        default=False,
        description="Whether to perform temporal alignment",
    )
    temporal_granularity: Optional[TemporalGranularity] = Field(
        None,
        description="Temporal granularity for alignment (if enabled)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, v: List[str]) -> List[str]:
        """Validate source_ids are non-empty strings and within MAX_SOURCES."""
        if len(v) > MAX_SOURCES:
            raise ValueError(
                f"source_ids cannot exceed {MAX_SOURCES} sources, got {len(v)}"
            )
        for sid in v:
            if not sid or not sid.strip():
                raise ValueError("source_ids must contain non-empty strings")
        return v


class BatchMatchResult(BaseModel):
    """Aggregated result of matching records across all source pairs.

    Provides counts, match rate, and the full list of individual
    match results for a batch matching operation.

    Attributes:
        total_pairs: Total number of record pairs evaluated.
        matched: Number of confidently matched pairs.
        unmatched: Number of unmatched records.
        ambiguous: Number of ambiguous match candidates.
        match_rate: Fraction of pairs that were matched (0.0-1.0).
        results: List of individual match results.
    """

    total_pairs: int = Field(
        default=0, ge=0,
        description="Total number of record pairs evaluated",
    )
    matched: int = Field(
        default=0, ge=0,
        description="Number of confidently matched pairs",
    )
    unmatched: int = Field(
        default=0, ge=0,
        description="Number of unmatched records",
    )
    ambiguous: int = Field(
        default=0, ge=0,
        description="Number of ambiguous match candidates",
    )
    match_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of pairs that were matched (0.0-1.0)",
    )
    results: List[MatchResult] = Field(
        default_factory=list,
        description="List of individual match results",
    )

    model_config = {"extra": "forbid"}


class DiscrepancySummary(BaseModel):
    """Aggregated summary of all discrepancies detected in a job.

    Breaks down discrepancies by type, severity, and source to
    support root-cause analysis and prioritization.

    Attributes:
        total: Total number of discrepancies detected.
        by_type: Count of discrepancies per discrepancy type.
        by_severity: Count of discrepancies per severity level.
        by_source: Count of discrepancies per source identifier.
        critical_count: Number of CRITICAL severity discrepancies.
        pending_review_count: Number of discrepancies pending review.
    """

    total: int = Field(
        default=0, ge=0,
        description="Total number of discrepancies detected",
    )
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of discrepancies per discrepancy type",
    )
    by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of discrepancies per severity level",
    )
    by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of discrepancies per source identifier",
    )
    critical_count: int = Field(
        default=0, ge=0,
        description="Number of CRITICAL severity discrepancies",
    )
    pending_review_count: int = Field(
        default=0, ge=0,
        description="Number of discrepancies pending review",
    )

    model_config = {"extra": "forbid"}


class ResolutionSummary(BaseModel):
    """Aggregated summary of all resolution decisions in a job.

    Provides breakdown by strategy, auto vs manual, and average
    confidence for monitoring resolution effectiveness.

    Attributes:
        total_resolved: Total number of discrepancies resolved.
        by_strategy: Count of resolutions per strategy.
        auto_resolved: Number of automatically resolved discrepancies.
        manual_resolved: Number of manually resolved discrepancies.
        pending: Number of discrepancies still pending resolution.
        average_confidence: Mean confidence across all resolutions (0.0-1.0).
    """

    total_resolved: int = Field(
        default=0, ge=0,
        description="Total number of discrepancies resolved",
    )
    by_strategy: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of resolutions per strategy",
    )
    auto_resolved: int = Field(
        default=0, ge=0,
        description="Number of automatically resolved discrepancies",
    )
    manual_resolved: int = Field(
        default=0, ge=0,
        description="Number of manually resolved discrepancies",
    )
    pending: int = Field(
        default=0, ge=0,
        description="Number of discrepancies still pending resolution",
    )
    average_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean confidence across all resolutions (0.0-1.0)",
    )

    model_config = {"extra": "forbid"}


class TemporalAlignment(BaseModel):
    """Result of aligning source records to a common temporal granularity.

    Reports the source and target granularities, aggregation method,
    and the number of records that were aligned or interpolated.

    Attributes:
        source_granularity: Original temporal granularity of the source data.
        target_granularity: Target temporal granularity after alignment.
        aggregation_method: Method used for temporal aggregation (e.g. 'sum', 'mean').
        records_aligned: Number of records successfully aligned.
        records_interpolated: Number of records created via interpolation.
    """

    source_granularity: TemporalGranularity = Field(
        ..., description="Original temporal granularity of the source data",
    )
    target_granularity: TemporalGranularity = Field(
        ..., description="Target temporal granularity after alignment",
    )
    aggregation_method: str = Field(
        default="sum",
        description="Method used for temporal aggregation (e.g. 'sum', 'mean')",
    )
    records_aligned: int = Field(
        default=0, ge=0,
        description="Number of records successfully aligned",
    )
    records_interpolated: int = Field(
        default=0, ge=0,
        description="Number of records created via interpolation",
    )

    model_config = {"extra": "forbid"}


class FieldLineage(BaseModel):
    """Lineage record for a single field in a golden record.

    Tracks the origin, original value, resolved value, and
    resolution strategy for one field in the golden record.

    Attributes:
        field_name: Name of the field.
        source_id: Identifier of the contributing source.
        source_name: Human-readable name of the contributing source.
        original_value: Original value from the contributing source.
        resolved_value: Final resolved value in the golden record.
        resolution_strategy: Strategy used to resolve this field.
        confidence: Confidence in the resolved value (0.0-1.0).
    """

    field_name: str = Field(
        ..., description="Name of the field",
    )
    source_id: str = Field(
        ..., description="Identifier of the contributing source",
    )
    source_name: str = Field(
        default="",
        description="Human-readable name of the contributing source",
    )
    original_value: Any = Field(
        default=None,
        description="Original value from the contributing source",
    )
    resolved_value: Any = Field(
        default=None,
        description="Final resolved value in the golden record",
    )
    resolution_strategy: str = Field(
        default="",
        description="Strategy used to resolve this field",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the resolved value (0.0-1.0)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v


class ReconciliationStats(BaseModel):
    """Aggregated operational statistics for the reconciliation service.

    Provides high-level metrics for monitoring overall health,
    throughput, and effectiveness of the reconciliation pipeline.

    Attributes:
        total_jobs: Total number of reconciliation jobs executed.
        total_sources: Total number of registered data sources.
        total_matches: Total record matches across all jobs.
        total_discrepancies: Total discrepancies detected across all jobs.
        total_golden_records: Total golden records assembled across all jobs.
        avg_match_confidence: Average match confidence across all jobs (0.0-1.0).
        avg_resolution_confidence: Average resolution confidence across all jobs (0.0-1.0).
    """

    total_jobs: int = Field(
        default=0, ge=0,
        description="Total number of reconciliation jobs executed",
    )
    total_sources: int = Field(
        default=0, ge=0,
        description="Total number of registered data sources",
    )
    total_matches: int = Field(
        default=0, ge=0,
        description="Total record matches across all jobs",
    )
    total_discrepancies: int = Field(
        default=0, ge=0,
        description="Total discrepancies detected across all jobs",
    )
    total_golden_records: int = Field(
        default=0, ge=0,
        description="Total golden records assembled across all jobs",
    )
    avg_match_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average match confidence across all jobs (0.0-1.0)",
    )
    avg_resolution_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average resolution confidence across all jobs (0.0-1.0)",
    )

    model_config = {"extra": "forbid"}


class SourceHealthMetrics(BaseModel):
    """Operational health metrics for a single data source.

    Tracks contribution volume, discrepancy rate, missing rate,
    average credibility, and last refresh timestamp.

    Attributes:
        source_id: Identifier of the monitored source.
        records_contributed: Total number of records from this source.
        discrepancy_rate: Fraction of records involved in discrepancies (0.0-1.0).
        missing_rate: Fraction of expected fields that are missing (0.0-1.0).
        avg_credibility: Average credibility score across assessments (0.0-1.0).
        last_refresh: Timestamp of the most recent data refresh.
    """

    source_id: str = Field(
        ..., description="Identifier of the monitored source",
    )
    records_contributed: int = Field(
        default=0, ge=0,
        description="Total number of records from this source",
    )
    discrepancy_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of records involved in discrepancies (0.0-1.0)",
    )
    missing_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of expected fields that are missing (0.0-1.0)",
    )
    avg_credibility: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average credibility score across assessments (0.0-1.0)",
    )
    last_refresh: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent data refresh",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v


class ComparisonSummary(BaseModel):
    """Aggregated summary of all field comparisons in a match batch.

    Provides counts of matches, mismatches, within-tolerance,
    missing, and incomparable fields along with the match rate.

    Attributes:
        total_fields_compared: Total number of field comparisons performed.
        matches: Number of exact field matches.
        mismatches: Number of field mismatches beyond tolerance.
        within_tolerance: Number of fields within tolerance.
        missing: Number of fields missing from one or both sides.
        incomparable: Number of fields that could not be compared.
        match_rate: Fraction of fields that matched (0.0-1.0).
    """

    total_fields_compared: int = Field(
        default=0, ge=0,
        description="Total number of field comparisons performed",
    )
    matches: int = Field(
        default=0, ge=0,
        description="Number of exact field matches",
    )
    mismatches: int = Field(
        default=0, ge=0,
        description="Number of field mismatches beyond tolerance",
    )
    within_tolerance: int = Field(
        default=0, ge=0,
        description="Number of fields within tolerance",
    )
    missing: int = Field(
        default=0, ge=0,
        description="Number of fields missing from one or both sides",
    )
    incomparable: int = Field(
        default=0, ge=0,
        description="Number of fields that could not be compared",
    )
    match_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of fields that matched (0.0-1.0)",
    )

    model_config = {"extra": "forbid"}


class PipelineStageResult(BaseModel):
    """Result of executing a single stage in the reconciliation pipeline.

    Records the stage name, status, record count, duration,
    errors, and provenance for pipeline observability.

    Attributes:
        stage_name: Name of the pipeline stage.
        status: Execution status of the stage.
        records_processed: Number of records processed in this stage.
        duration_ms: Stage execution duration in milliseconds.
        errors: List of error messages from this stage.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    stage_name: str = Field(
        ..., description="Name of the pipeline stage",
    )
    status: ReconciliationStatus = Field(
        default=ReconciliationStatus.PENDING,
        description="Execution status of the stage",
    )
    records_processed: int = Field(
        default=0, ge=0,
        description="Number of records processed in this stage",
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0,
        description="Stage execution duration in milliseconds",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages from this stage",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("stage_name")
    @classmethod
    def validate_stage_name(cls, v: str) -> str:
        """Validate stage_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("stage_name must be non-empty")
        return v


class ReconciliationEvent(BaseModel):
    """An audit event emitted during reconciliation pipeline execution.

    Records a timestamped event with type, job context, and
    arbitrary detail payload for observability and audit.

    Attributes:
        event_id: Unique identifier for this event.
        job_id: Identifier of the parent reconciliation job.
        event_type: Type of the event (e.g. 'match_started', 'discrepancy_found').
        timestamp: When the event occurred.
        details: Arbitrary event detail payload.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this event",
    )
    job_id: str = Field(
        ..., description="Identifier of the parent reconciliation job",
    )
    event_type: str = Field(
        ..., description="Type of the event (e.g. 'match_started', 'discrepancy_found')",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="When the event occurred",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary event detail payload",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("job_id must be non-empty")
        return v

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("event_type must be non-empty")
        return v


# =============================================================================
# Request Models (8)
# =============================================================================


class CreateJobRequest(BaseModel):
    """Request body for creating a new reconciliation job.

    Attributes:
        job_name: Human-readable name for the reconciliation job.
        source_ids: List of source identifiers to include (min 2).
        match_strategy: Record matching strategy to use.
        resolution_strategy: Discrepancy resolution strategy to use.
        tolerance_rules: List of field-level tolerance rules.
        enable_golden_records: Whether to assemble golden records.
        enable_temporal_alignment: Whether to perform temporal alignment.
        temporal_granularity: Temporal granularity for alignment (if enabled).
    """

    job_name: str = Field(
        default="",
        description="Human-readable name for the reconciliation job",
    )
    source_ids: List[str] = Field(
        ..., min_length=2,
        description="List of source identifiers to include",
    )
    match_strategy: MatchStrategy = Field(
        default=MatchStrategy.COMPOSITE,
        description="Record matching strategy to use",
    )
    resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.PRIORITY_WINS,
        description="Discrepancy resolution strategy to use",
    )
    tolerance_rules: List[ToleranceRule] = Field(
        default_factory=list,
        description="List of field-level tolerance rules",
    )
    enable_golden_records: bool = Field(
        default=True,
        description="Whether to assemble golden records",
    )
    enable_temporal_alignment: bool = Field(
        default=False,
        description="Whether to perform temporal alignment",
    )
    temporal_granularity: Optional[TemporalGranularity] = Field(
        None,
        description="Temporal granularity for alignment (if enabled)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, v: List[str]) -> List[str]:
        """Validate source_ids are non-empty strings and within MAX_SOURCES."""
        if len(v) > MAX_SOURCES:
            raise ValueError(
                f"source_ids cannot exceed {MAX_SOURCES} sources, got {len(v)}"
            )
        for sid in v:
            if not sid or not sid.strip():
                raise ValueError("source_ids must contain non-empty strings")
        return v


class RegisterSourceRequest(BaseModel):
    """Request body for registering a new data source.

    Attributes:
        name: Human-readable source name.
        source_type: Classification of the source system.
        priority: Source priority for conflict resolution (1-100).
        schema_info: Schema metadata describing the source's field structure.
        refresh_cadence: How frequently the source refreshes.
        description: Human-readable description of the source.
        tags: List of tags for categorization and filtering.
    """

    name: str = Field(
        ..., description="Human-readable source name",
    )
    source_type: SourceType = Field(
        default=SourceType.OTHER,
        description="Classification of the source system",
    )
    priority: int = Field(
        default=50, ge=1, le=100,
        description="Source priority for conflict resolution (1=lowest, 100=highest)",
    )
    schema_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema metadata describing the source's field structure",
    )
    refresh_cadence: str = Field(
        default="daily",
        description="How frequently the source refreshes",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the source",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorization and filtering",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class UpdateSourceRequest(BaseModel):
    """Request body for updating an existing data source.

    All fields are optional; only provided fields will be updated.

    Attributes:
        name: Updated human-readable source name.
        priority: Updated source priority (1-100).
        credibility_score: Updated credibility score (0.0-1.0).
        status: Updated operational status.
        schema_info: Updated schema metadata.
    """

    name: Optional[str] = Field(
        None,
        description="Updated human-readable source name",
    )
    priority: Optional[int] = Field(
        None, ge=1, le=100,
        description="Updated source priority (1-100)",
    )
    credibility_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Updated credibility score (0.0-1.0)",
    )
    status: Optional[SourceStatus] = Field(
        None,
        description="Updated operational status",
    )
    schema_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated schema metadata",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate name is non-empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("name must be non-empty when provided")
        return v


class MatchRequest(BaseModel):
    """Request body for executing a record matching operation.

    Attributes:
        source_ids: List of source identifiers to match across.
        match_strategy: Matching strategy to use.
        match_threshold: Minimum confidence threshold for accepting a match (0.0-1.0).
        key_fields: List of field names to use as match keys.
        enable_fuzzy: Whether to enable fuzzy matching.
    """

    source_ids: List[str] = Field(
        ..., min_length=2,
        description="List of source identifiers to match across",
    )
    match_strategy: MatchStrategy = Field(
        default=MatchStrategy.COMPOSITE,
        description="Matching strategy to use",
    )
    match_threshold: float = Field(
        default=DEFAULT_MATCH_THRESHOLD, ge=0.0, le=1.0,
        description="Minimum confidence threshold for accepting a match (0.0-1.0)",
    )
    key_fields: List[str] = Field(
        default_factory=list,
        description="List of field names to use as match keys",
    )
    enable_fuzzy: bool = Field(
        default=False,
        description="Whether to enable fuzzy matching",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, v: List[str]) -> List[str]:
        """Validate source_ids are non-empty strings."""
        for sid in v:
            if not sid or not sid.strip():
                raise ValueError("source_ids must contain non-empty strings")
        return v


class CompareRequest(BaseModel):
    """Request body for comparing fields across matched record pairs.

    Attributes:
        match_ids: List of match result identifiers to compare.
        tolerance_rules: List of field-level tolerance rules.
        field_types: Mapping of field names to their data types.
    """

    match_ids: List[str] = Field(
        ..., min_length=1,
        description="List of match result identifiers to compare",
    )
    tolerance_rules: List[ToleranceRule] = Field(
        default_factory=list,
        description="List of field-level tolerance rules",
    )
    field_types: Dict[str, FieldType] = Field(
        default_factory=dict,
        description="Mapping of field names to their data types",
    )

    model_config = {"extra": "forbid"}

    @field_validator("match_ids")
    @classmethod
    def validate_match_ids(cls, v: List[str]) -> List[str]:
        """Validate match_ids are non-empty strings."""
        for mid in v:
            if not mid or not mid.strip():
                raise ValueError("match_ids must contain non-empty strings")
        return v


class ResolveRequest(BaseModel):
    """Request body for resolving detected discrepancies.

    Attributes:
        discrepancy_ids: List of discrepancy identifiers to resolve.
        strategy: Resolution strategy to apply.
        manual_values: Optional mapping of discrepancy_id to manual override values.
    """

    discrepancy_ids: List[str] = Field(
        ..., min_length=1,
        description="List of discrepancy identifiers to resolve",
    )
    strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.PRIORITY_WINS,
        description="Resolution strategy to apply",
    )
    manual_values: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional mapping of discrepancy_id to manual override values",
    )

    model_config = {"extra": "forbid"}

    @field_validator("discrepancy_ids")
    @classmethod
    def validate_discrepancy_ids(cls, v: List[str]) -> List[str]:
        """Validate discrepancy_ids are non-empty strings."""
        for did in v:
            if not did or not did.strip():
                raise ValueError("discrepancy_ids must contain non-empty strings")
        return v


class PipelineRequest(BaseModel):
    """Request body for running the full reconciliation pipeline.

    Triggers end-to-end matching, comparison, discrepancy detection,
    resolution, and optional golden record assembly.

    Attributes:
        source_ids: List of source identifiers to reconcile.
        match_strategy: Record matching strategy to use.
        resolution_strategy: Discrepancy resolution strategy to use.
        tolerance_rules: List of field-level tolerance rules.
        enable_golden_records: Whether to assemble golden records.
    """

    source_ids: List[str] = Field(
        ..., min_length=2,
        description="List of source identifiers to reconcile",
    )
    match_strategy: MatchStrategy = Field(
        default=MatchStrategy.COMPOSITE,
        description="Record matching strategy to use",
    )
    resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.PRIORITY_WINS,
        description="Discrepancy resolution strategy to use",
    )
    tolerance_rules: List[ToleranceRule] = Field(
        default_factory=list,
        description="List of field-level tolerance rules",
    )
    enable_golden_records: bool = Field(
        default=True,
        description="Whether to assemble golden records",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, v: List[str]) -> List[str]:
        """Validate source_ids are non-empty strings and within MAX_SOURCES."""
        if len(v) > MAX_SOURCES:
            raise ValueError(
                f"source_ids cannot exceed {MAX_SOURCES} sources, got {len(v)}"
            )
        for sid in v:
            if not sid or not sid.strip():
                raise ValueError("source_ids must contain non-empty strings")
        return v


class GoldenRecordRequest(BaseModel):
    """Request body for assembling a golden record for a specific entity.

    Attributes:
        entity_id: Identifier of the entity to build a golden record for.
        period: Reporting period for the golden record.
        source_priority_overrides: Optional mapping of source_id to priority overrides.
    """

    entity_id: str = Field(
        ..., description="Identifier of the entity to build a golden record for",
    )
    period: str = Field(
        ..., description="Reporting period for the golden record",
    )
    source_priority_overrides: Optional[Dict[str, int]] = Field(
        None,
        description="Optional mapping of source_id to priority overrides",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        """Validate period is non-empty."""
        if not v or not v.strip():
            raise ValueError("period must be non-empty")
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (5)
    # -------------------------------------------------------------------------
    "ConsistencyAnalyzer",
    "SimilarityScorer",
    "MatchClassifier",
    "FactorReconciler",
    "ConflictResolutionStrategy",
    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    "_utcnow",
    # -------------------------------------------------------------------------
    # Constants (10)
    # -------------------------------------------------------------------------
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_TOLERANCE_PCT",
    "DEFAULT_TOLERANCE_ABS",
    "MAX_SOURCES",
    "MAX_MATCH_CANDIDATES",
    "CRITICAL_THRESHOLD_PCT",
    "HIGH_THRESHOLD_PCT",
    "MEDIUM_THRESHOLD_PCT",
    "SUPPORTED_UNITS",
    "SUPPORTED_CURRENCIES",
    # -------------------------------------------------------------------------
    # Enumerations (13)
    # -------------------------------------------------------------------------
    "SourceType",
    "SourceStatus",
    "MatchStrategy",
    "MatchStatus",
    "ComparisonResult",
    "DiscrepancyType",
    "DiscrepancySeverity",
    "ResolutionStrategy",
    "ResolutionStatus",
    "FieldType",
    "ReconciliationStatus",
    "TemporalGranularity",
    "CredibilityFactor",
    # -------------------------------------------------------------------------
    # SDK data models (22)
    # -------------------------------------------------------------------------
    "SourceDefinition",
    "SchemaMapping",
    "MatchKey",
    "MatchResult",
    "FieldComparison",
    "Discrepancy",
    "ResolutionDecision",
    "GoldenRecord",
    "SourceCredibility",
    "ToleranceRule",
    "ReconciliationReport",
    "ReconciliationJobConfig",
    "BatchMatchResult",
    "DiscrepancySummary",
    "ResolutionSummary",
    "TemporalAlignment",
    "FieldLineage",
    "ReconciliationStats",
    "SourceHealthMetrics",
    "ComparisonSummary",
    "PipelineStageResult",
    "ReconciliationEvent",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "CreateJobRequest",
    "RegisterSourceRequest",
    "UpdateSourceRequest",
    "MatchRequest",
    "CompareRequest",
    "ResolveRequest",
    "PipelineRequest",
    "GoldenRecordRequest",
]
