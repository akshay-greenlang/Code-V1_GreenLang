# -*- coding: utf-8 -*-
"""
Duplicate Detection Agent Service Data Models - AGENT-DATA-011

Pydantic v2 data models for the Duplicate Detection SDK. Provides
comprehensive type-safe models for record fingerprinting, blocking,
pairwise comparison, match classification, duplicate clustering,
record merging, dedup rule definition, job management, pipeline
checkpointing, statistics, and reporting.

New enumerations (13):
    - FingerprintAlgorithm, BlockingStrategy, SimilarityAlgorithm,
      MatchClassification, ClusterAlgorithm, MergeStrategy,
      ConflictResolution, JobStatus, PipelineStage, RecordSource,
      FieldType, IssueSeverity, ReportFormat

New SDK models (15):
    - RecordFingerprint, BlockResult, SimilarityResult,
      FieldComparisonConfig, MatchResult, DuplicateCluster,
      MergeDecision, MergeConflict, DedupRule, DedupJob,
      PipelineCheckpoint, DedupStatistics, DedupReport,
      DedupIssueSummary, DedupJobSummary

Request models (7):
    - FingerprintRequest, BlockRequest, CompareRequest,
      ClassifyRequest, ClusterRequest, MergeRequest,
      PipelineRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default field weights for common sustainability dedup fields.
DEFAULT_FIELD_WEIGHTS: Dict[str, float] = {
    "name": 0.25,
    "address": 0.20,
    "email": 0.15,
    "phone": 0.10,
    "company": 0.10,
    "identifier": 0.10,
    "date": 0.05,
    "amount": 0.05,
}

#: Default configuration per similarity algorithm.
SIMILARITY_ALGORITHM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "exact": {"case_sensitive": True},
    "levenshtein": {"max_distance": 5, "normalize": True},
    "jaro_winkler": {"winkler_prefix_weight": 0.1, "long_tolerance": False},
    "soundex": {"length": 4},
    "ngram": {"n": 3, "padded": True},
    "tfidf_cosine": {"min_df": 1, "max_df": 1.0, "ngram_range": [1, 2]},
    "numeric": {"tolerance_pct": 0.01, "absolute_tolerance": 0.0},
    "date": {"tolerance_days": 1, "format": "ISO-8601"},
}

#: Default configuration per blocking strategy.
BLOCKING_STRATEGY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sorted_neighborhood": {"window_size": 10, "key_size": 3},
    "standard": {"key_fields": []},
    "canopy": {"tight_threshold": 0.8, "loose_threshold": 0.4},
    "none": {},
}

#: Maximum number of records allowed in a single comparison block.
MAX_RECORDS_PER_COMPARISON: int = 50_000

#: Default n-gram size for character-level similarity.
DEFAULT_NGRAM_SIZE: int = 3

#: Default match threshold for duplicate classification.
DEFAULT_MATCH_THRESHOLD: float = 0.85

#: Default possible-match threshold.
DEFAULT_POSSIBLE_THRESHOLD: float = 0.65

#: Default non-match threshold.
DEFAULT_NON_MATCH_THRESHOLD: float = 0.40

#: Maximum pairwise comparisons before block truncation warning.
MAX_BLOCK_COMPARISONS_WARN: int = 100_000

#: All supported fingerprint algorithms.
FINGERPRINT_ALGORITHMS: tuple = ("sha256", "simhash", "minhash")

#: All supported blocking strategies.
BLOCKING_STRATEGIES: tuple = ("sorted_neighborhood", "standard", "canopy", "none")

#: All supported similarity algorithms.
SIMILARITY_ALGORITHMS: tuple = (
    "exact", "levenshtein", "jaro_winkler", "soundex",
    "ngram", "tfidf_cosine", "numeric", "date",
)

#: All supported merge strategies.
MERGE_STRATEGIES: tuple = (
    "keep_first", "keep_latest", "keep_most_complete",
    "merge_fields", "golden_record", "custom",
)

#: All supported conflict resolution methods.
CONFLICT_RESOLUTIONS: tuple = (
    "first", "latest", "most_complete", "longest", "shortest",
)

#: Pipeline stage execution order.
PIPELINE_STAGE_ORDER: tuple = (
    "fingerprint", "block", "compare", "classify",
    "cluster", "merge", "complete",
)

#: All supported field types for comparison configuration.
FIELD_TYPES: tuple = ("string", "numeric", "date", "boolean", "categorical")

#: Issue severity levels ordered by increasing severity.
ISSUE_SEVERITY_ORDER: tuple = ("info", "low", "medium", "high", "critical")

#: Report format options.
REPORT_FORMAT_OPTIONS: tuple = ("json", "markdown", "html", "text", "csv")


# =============================================================================
# Enumerations (13)
# =============================================================================


class FingerprintAlgorithm(str, Enum):
    """Algorithm used for generating record fingerprints.

    SHA256 produces deterministic cryptographic hashes for exact-match
    deduplication. SIMHASH produces locality-sensitive hashes where
    similar records hash to similar values. MINHASH approximates
    Jaccard similarity via random permutation signatures.
    """

    SHA256 = "sha256"
    SIMHASH = "simhash"
    MINHASH = "minhash"


class BlockingStrategy(str, Enum):
    """Strategy for generating candidate record pairs.

    Blocking reduces the quadratic comparison space by grouping
    records into blocks where only within-block comparisons are
    performed.

    SORTED_NEIGHBORHOOD: Sort by blocking key, slide window.
    STANDARD: Exact match on blocking key fields.
    CANOPY: Distance-based overlapping canopies.
    NONE: Compare all pairs (use only for small datasets).
    """

    SORTED_NEIGHBORHOOD = "sorted_neighborhood"
    STANDARD = "standard"
    CANOPY = "canopy"
    NONE = "none"


class SimilarityAlgorithm(str, Enum):
    """Algorithm for computing field-level similarity scores.

    Each algorithm is optimized for a specific field type or
    comparison requirement. All algorithms produce scores in
    the range [0.0, 1.0] where 1.0 indicates exact match.

    EXACT: Binary exact match (0.0 or 1.0).
    LEVENSHTEIN: Edit distance normalized by max string length.
    JARO_WINKLER: Jaro-Winkler distance for short strings/names.
    SOUNDEX: Phonetic encoding comparison for names.
    NGRAM: Character n-gram overlap (Dice/Jaccard coefficient).
    TFIDF_COSINE: TF-IDF weighted cosine similarity for text.
    NUMERIC: Proportional distance for numeric fields.
    DATE: Temporal proximity for date/datetime fields.
    """

    EXACT = "exact"
    LEVENSHTEIN = "levenshtein"
    JARO_WINKLER = "jaro_winkler"
    SOUNDEX = "soundex"
    NGRAM = "ngram"
    TFIDF_COSINE = "tfidf_cosine"
    NUMERIC = "numeric"
    DATE = "date"


class MatchClassification(str, Enum):
    """Classification outcome of a record pair comparison.

    Based on the Fellegi-Sunter decision model with three regions:
    MATCH (above upper threshold), NON_MATCH (below lower threshold),
    and POSSIBLE (between thresholds, requires manual review).
    """

    MATCH = "match"
    NON_MATCH = "non_match"
    POSSIBLE = "possible"


class ClusterAlgorithm(str, Enum):
    """Algorithm for grouping matched pairs into duplicate clusters.

    UNION_FIND: Disjoint-set data structure with path compression
        and union by rank. O(alpha(n)) amortized per operation.
    CONNECTED_COMPONENTS: Graph-based BFS/DFS connected component
        discovery. Better for sparse match graphs.
    """

    UNION_FIND = "union_find"
    CONNECTED_COMPONENTS = "connected_components"


class MergeStrategy(str, Enum):
    """Strategy for merging duplicate records within a cluster.

    Determines which record or field values survive after
    deduplication.

    KEEP_FIRST: Keep the first record encountered (by insertion order).
    KEEP_LATEST: Keep the most recently updated record.
    KEEP_MOST_COMPLETE: Keep the record with the fewest null fields.
    MERGE_FIELDS: Combine non-null fields from all records.
    GOLDEN_RECORD: Create a synthesized best-of-breed record using
        field-level quality scores.
    CUSTOM: Apply a user-defined merge function.
    """

    KEEP_FIRST = "keep_first"
    KEEP_LATEST = "keep_latest"
    KEEP_MOST_COMPLETE = "keep_most_complete"
    MERGE_FIELDS = "merge_fields"
    GOLDEN_RECORD = "golden_record"
    CUSTOM = "custom"


class ConflictResolution(str, Enum):
    """Method for resolving field-level conflicts during record merge.

    When two or more source records disagree on a field value,
    this method determines which value is used in the merged output.

    FIRST: Use the value from the first record (by insertion order).
    LATEST: Use the value from the most recently updated record.
    MOST_COMPLETE: Use the value from the record with fewest nulls.
    LONGEST: Use the longest non-null string value.
    SHORTEST: Use the shortest non-null string value.
    """

    FIRST = "first"
    LATEST = "latest"
    MOST_COMPLETE = "most_complete"
    LONGEST = "longest"
    SHORTEST = "shortest"


class JobStatus(str, Enum):
    """Lifecycle status of a deduplication job.

    Tracks the current execution state of a dedup pipeline
    from submission through completion, failure, or cancellation.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    """Stage in the deduplication pipeline.

    The pipeline executes stages in order: fingerprint, block,
    compare, classify, cluster, merge, complete. Each stage
    produces output consumed by the next stage.
    """

    FINGERPRINT = "fingerprint"
    BLOCK = "block"
    COMPARE = "compare"
    CLASSIFY = "classify"
    CLUSTER = "cluster"
    MERGE = "merge"
    COMPLETE = "complete"


class RecordSource(str, Enum):
    """Source mode for record deduplication.

    SINGLE_DATASET: Deduplicate records within a single dataset
        (entity resolution / dedup).
    CROSS_DATASET: Link records across two or more datasets
        (record linkage / matching).
    """

    SINGLE_DATASET = "single_dataset"
    CROSS_DATASET = "cross_dataset"


class FieldType(str, Enum):
    """Data type classification for comparison field configuration.

    Determines which similarity algorithms are applicable and
    how normalization is performed before comparison.
    """

    STRING = "string"
    NUMERIC = "numeric"
    DATE = "date"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class IssueSeverity(str, Enum):
    """Severity classification for deduplication issues.

    Determines the visibility and required action for each
    issue detected during the dedup pipeline.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReportFormat(str, Enum):
    """Output format for deduplication reports.

    Defines the serialization format for generated reports
    including summaries, cluster details, and merge decisions.
    """

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


# =============================================================================
# SDK Data Models (15)
# =============================================================================


class RecordFingerprint(BaseModel):
    """Fingerprint computed from one or more fields of a record.

    Record fingerprints enable fast exact-match detection and serve
    as blocking keys for candidate pair generation. The fingerprint
    hash is deterministic given the same input fields, algorithm,
    and normalization settings.

    Attributes:
        record_id: Unique identifier of the source record.
        dataset_id: Identifier of the dataset the record belongs to.
        field_set: List of field names used to compute the fingerprint.
        fingerprint_hash: Computed fingerprint hash value.
        algorithm: Algorithm used to compute the fingerprint.
        normalized_fields: Whether fields were normalized before hashing.
        created_at: Timestamp when the fingerprint was computed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_id: str = Field(
        ..., description="Unique identifier of the source record",
    )
    dataset_id: str = Field(
        default="", description="Identifier of the dataset the record belongs to",
    )
    field_set: List[str] = Field(
        default_factory=list,
        description="List of field names used to compute the fingerprint",
    )
    fingerprint_hash: str = Field(
        default="",
        description="Computed fingerprint hash value",
    )
    algorithm: FingerprintAlgorithm = Field(
        default=FingerprintAlgorithm.SHA256,
        description="Algorithm used to compute the fingerprint",
    )
    normalized_fields: bool = Field(
        default=True,
        description="Whether fields were normalized before hashing",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the fingerprint was computed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, v: str) -> str:
        """Validate record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_id must be non-empty")
        return v


class BlockResult(BaseModel):
    """Result of a blocking operation for candidate pair generation.

    Contains the blocking key, the strategy used, and the list of
    record identifiers within this block. Only records within the
    same block are compared pairwise.

    Attributes:
        block_key: The computed blocking key for this group.
        strategy: Blocking strategy that produced this block.
        record_ids: List of record identifiers in this block.
        record_count: Number of records in this block.
        created_at: Timestamp when the block was created.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    block_key: str = Field(
        ..., description="The computed blocking key for this group",
    )
    strategy: BlockingStrategy = Field(
        default=BlockingStrategy.SORTED_NEIGHBORHOOD,
        description="Blocking strategy that produced this block",
    )
    record_ids: List[str] = Field(
        default_factory=list,
        description="List of record identifiers in this block",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Number of records in this block",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the block was created",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("block_key")
    @classmethod
    def validate_block_key(cls, v: str) -> str:
        """Validate block_key is non-empty."""
        if not v or not v.strip():
            raise ValueError("block_key must be non-empty")
        return v


class SimilarityResult(BaseModel):
    """Result of pairwise field-level similarity comparison.

    Contains per-field similarity scores, the weighted overall
    score, algorithm metadata, and comparison timing. This is
    the raw comparison output before classification.

    Attributes:
        record_a_id: Identifier of the first record in the pair.
        record_b_id: Identifier of the second record in the pair.
        field_scores: Per-field similarity scores (field_name -> score).
        overall_score: Weighted overall similarity score (0.0 to 1.0).
        algorithm_used: Primary similarity algorithm used.
        comparison_time_ms: Time in milliseconds to compute this comparison.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_a_id: str = Field(
        ..., description="Identifier of the first record in the pair",
    )
    record_b_id: str = Field(
        ..., description="Identifier of the second record in the pair",
    )
    field_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field similarity scores (field_name -> score 0.0-1.0)",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall similarity score (0.0 to 1.0)",
    )
    algorithm_used: SimilarityAlgorithm = Field(
        default=SimilarityAlgorithm.JARO_WINKLER,
        description="Primary similarity algorithm used",
    )
    comparison_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Time in milliseconds to compute this comparison",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_a_id")
    @classmethod
    def validate_record_a_id(cls, v: str) -> str:
        """Validate record_a_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_a_id must be non-empty")
        return v

    @field_validator("record_b_id")
    @classmethod
    def validate_record_b_id(cls, v: str) -> str:
        """Validate record_b_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_b_id must be non-empty")
        return v


class FieldComparisonConfig(BaseModel):
    """Configuration for comparing a specific field across record pairs.

    Defines the similarity algorithm, weight, field type, and
    preprocessing options for a single field in the comparison
    rule set.

    Attributes:
        field_name: Name of the field to compare.
        algorithm: Similarity algorithm to use for this field.
        weight: Weight of this field in the overall similarity score.
        field_type: Data type classification for this field.
        case_sensitive: Whether comparison is case-sensitive.
        strip_whitespace: Whether to strip leading/trailing whitespace.
        phonetic_encode: Whether to apply phonetic encoding before
            comparison (Soundex, Metaphone).
    """

    field_name: str = Field(
        ..., description="Name of the field to compare",
    )
    algorithm: SimilarityAlgorithm = Field(
        default=SimilarityAlgorithm.JARO_WINKLER,
        description="Similarity algorithm to use for this field",
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Weight of this field in the overall similarity score",
    )
    field_type: FieldType = Field(
        default=FieldType.STRING,
        description="Data type classification for this field",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether comparison is case-sensitive",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip leading/trailing whitespace",
    )
    phonetic_encode: bool = Field(
        default=False,
        description="Whether to apply phonetic encoding before comparison",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class MatchResult(BaseModel):
    """Classification result for a record pair comparison.

    Combines the similarity scores with classification decision,
    confidence level, and decision reasoning. This is the output
    of the classification stage.

    Attributes:
        record_a_id: Identifier of the first record in the pair.
        record_b_id: Identifier of the second record in the pair.
        classification: Match/non-match/possible classification.
        confidence: Confidence in the classification (0.0 to 1.0).
        field_scores: Per-field similarity scores from comparison.
        overall_score: Weighted overall similarity score.
        decision_reason: Human-readable explanation of the classification.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_a_id: str = Field(
        ..., description="Identifier of the first record in the pair",
    )
    record_b_id: str = Field(
        ..., description="Identifier of the second record in the pair",
    )
    classification: MatchClassification = Field(
        ..., description="Match/non-match/possible classification",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the classification (0.0 to 1.0)",
    )
    field_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field similarity scores from comparison",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall similarity score",
    )
    decision_reason: str = Field(
        default="",
        description="Human-readable explanation of the classification",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_a_id")
    @classmethod
    def validate_record_a_id(cls, v: str) -> str:
        """Validate record_a_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_a_id must be non-empty")
        return v

    @field_validator("record_b_id")
    @classmethod
    def validate_record_b_id(cls, v: str) -> str:
        """Validate record_b_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_b_id must be non-empty")
        return v


class MergeConflict(BaseModel):
    """A field-level conflict encountered during record merge.

    Represents a disagreement between source records on a specific
    field value, including all competing values, the chosen winner,
    and the resolution method applied.

    Attributes:
        field_name: Name of the field with conflicting values.
        values: Mapping of record_id to field value for all sources.
        chosen_value: The value chosen after conflict resolution.
        resolution_method: The conflict resolution method applied.
        source_record_id: Identifier of the record that supplied
            the chosen value (if applicable).
    """

    field_name: str = Field(
        ..., description="Name of the field with conflicting values",
    )
    values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of record_id to field value for all sources",
    )
    chosen_value: Optional[Any] = Field(
        None, description="The value chosen after conflict resolution",
    )
    resolution_method: ConflictResolution = Field(
        default=ConflictResolution.MOST_COMPLETE,
        description="The conflict resolution method applied",
    )
    source_record_id: Optional[str] = Field(
        None,
        description="Identifier of the record that supplied the chosen value",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class DuplicateCluster(BaseModel):
    """A cluster of duplicate records identified during deduplication.

    Groups record identifiers that have been transitively linked
    through pairwise match decisions. Includes quality metrics
    and a designated representative record.

    Attributes:
        cluster_id: Unique identifier for this duplicate cluster.
        member_record_ids: List of record identifiers in the cluster.
        representative_id: Identifier of the representative record
            (e.g., the most complete or most recent).
        cluster_quality: Overall quality score of the cluster
            (average pairwise similarity, 0.0 to 1.0).
        density: Ratio of actual edges to possible edges in the
            cluster match graph (0.0 to 1.0).
        diameter: Maximum pairwise distance within the cluster.
        member_count: Number of records in the cluster.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    cluster_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this duplicate cluster",
    )
    member_record_ids: List[str] = Field(
        default_factory=list,
        description="List of record identifiers in the cluster",
    )
    representative_id: Optional[str] = Field(
        None,
        description="Identifier of the representative record",
    )
    cluster_quality: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall quality score of the cluster (0.0 to 1.0)",
    )
    density: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Ratio of actual edges to possible edges (0.0 to 1.0)",
    )
    diameter: float = Field(
        default=0.0, ge=0.0,
        description="Maximum pairwise distance within the cluster",
    )
    member_count: int = Field(
        default=0, ge=0,
        description="Number of records in the cluster",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("member_record_ids")
    @classmethod
    def validate_member_record_ids(cls, v: List[str]) -> List[str]:
        """Validate member_record_ids contains no empty strings."""
        for rid in v:
            if not rid or not rid.strip():
                raise ValueError("member_record_ids must not contain empty strings")
        return v


class MergeDecision(BaseModel):
    """Decision record for merging a duplicate cluster into one record.

    Contains the merge strategy used, the merged output record,
    any field-level conflicts encountered, source record details,
    and full provenance for audit trail.

    Attributes:
        cluster_id: Identifier of the cluster being merged.
        strategy: Merge strategy applied.
        merged_record: The merged output record as a dictionary.
        conflicts: List of field-level conflicts encountered.
        source_records: List of source record dictionaries.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        decided_at: Timestamp when the merge decision was made.
    """

    cluster_id: str = Field(
        ..., description="Identifier of the cluster being merged",
    )
    strategy: MergeStrategy = Field(
        default=MergeStrategy.KEEP_MOST_COMPLETE,
        description="Merge strategy applied",
    )
    merged_record: Dict[str, Any] = Field(
        default_factory=dict,
        description="The merged output record as a dictionary",
    )
    conflicts: List[MergeConflict] = Field(
        default_factory=list,
        description="List of field-level conflicts encountered",
    )
    source_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of source record dictionaries",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    decided_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the merge decision was made",
    )

    model_config = {"extra": "forbid"}

    @field_validator("cluster_id")
    @classmethod
    def validate_cluster_id(cls, v: str) -> str:
        """Validate cluster_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("cluster_id must be non-empty")
        return v


class DedupRule(BaseModel):
    """A deduplication rule defining how records are compared and matched.

    Encapsulates the full comparison configuration including field-level
    comparison settings, thresholds, blocking strategy, and merge
    strategy. Rules are reusable across multiple dedup jobs.

    Attributes:
        rule_id: Unique identifier for this dedup rule.
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        field_configs: List of field comparison configurations.
        match_threshold: Overall score threshold for MATCH classification.
        possible_threshold: Overall score threshold for POSSIBLE
            classification.
        blocking_strategy: Blocking strategy for candidate generation.
        blocking_fields: List of field names used for blocking keys.
        merge_strategy: Strategy for merging duplicate clusters.
        active: Whether this rule is currently active.
        created_at: Timestamp when the rule was created.
        updated_at: Timestamp when the rule was last updated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this dedup rule",
    )
    name: str = Field(
        ..., description="Human-readable rule name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the rule purpose",
    )
    field_configs: List[FieldComparisonConfig] = Field(
        default_factory=list,
        description="List of field comparison configurations",
    )
    match_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Overall score threshold for MATCH classification",
    )
    possible_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="Overall score threshold for POSSIBLE classification",
    )
    blocking_strategy: BlockingStrategy = Field(
        default=BlockingStrategy.SORTED_NEIGHBORHOOD,
        description="Blocking strategy for candidate generation",
    )
    blocking_fields: List[str] = Field(
        default_factory=list,
        description="List of field names used for blocking keys",
    )
    merge_strategy: MergeStrategy = Field(
        default=MergeStrategy.KEEP_MOST_COMPLETE,
        description="Strategy for merging duplicate clusters",
    )
    active: bool = Field(
        default=True,
        description="Whether this rule is currently active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the rule was created",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the rule was last updated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("match_threshold")
    @classmethod
    def validate_match_threshold(cls, v: float) -> float:
        """Validate match_threshold is reasonable."""
        if v < 0.5:
            raise ValueError(
                "match_threshold must be >= 0.5 to avoid excessive false positives"
            )
        return v

    @field_validator("possible_threshold")
    @classmethod
    def validate_possible_threshold(cls, v: float) -> float:
        """Validate possible_threshold is reasonable."""
        if v < 0.1:
            raise ValueError(
                "possible_threshold must be >= 0.1 to be meaningful"
            )
        return v


class DedupJob(BaseModel):
    """A deduplication job tracking the full pipeline execution.

    Represents a single end-to-end dedup run from fingerprinting
    through merge, with progress counters, timing, error tracking,
    and provenance for audit trail.

    Attributes:
        job_id: Unique identifier for this dedup job.
        dataset_ids: List of dataset identifiers being deduplicated.
        rule_id: Identifier of the dedup rule applied.
        status: Current job execution status.
        stage: Current pipeline stage being executed.
        total_records: Total number of input records.
        fingerprinted: Number of records fingerprinted so far.
        compared: Number of pairwise comparisons completed.
        matched: Number of record pairs classified as MATCH.
        clustered: Number of duplicate clusters formed.
        merged: Number of clusters merged into single records.
        error_message: Error message if the job failed.
        started_at: Timestamp when the job started.
        completed_at: Timestamp when the job completed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this dedup job",
    )
    dataset_ids: List[str] = Field(
        default_factory=list,
        description="List of dataset identifiers being deduplicated",
    )
    rule_id: Optional[str] = Field(
        None,
        description="Identifier of the dedup rule applied",
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current job execution status",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.FINGERPRINT,
        description="Current pipeline stage being executed",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of input records",
    )
    fingerprinted: int = Field(
        default=0, ge=0,
        description="Number of records fingerprinted so far",
    )
    compared: int = Field(
        default=0, ge=0,
        description="Number of pairwise comparisons completed",
    )
    matched: int = Field(
        default=0, ge=0,
        description="Number of record pairs classified as MATCH",
    )
    clustered: int = Field(
        default=0, ge=0,
        description="Number of duplicate clusters formed",
    )
    merged: int = Field(
        default=0, ge=0,
        description="Number of clusters merged into single records",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if the job failed",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the job started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the job completed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @property
    def is_active(self) -> bool:
        """Return True if the job is currently executing."""
        return self.status in (JobStatus.PENDING, JobStatus.RUNNING)

    @property
    def progress_pct(self) -> float:
        """Return pipeline progress as a percentage (0.0 to 100.0).

        Maps the current stage to a progress value based on
        pipeline stage order.
        """
        stage_progress = {
            PipelineStage.FINGERPRINT: 10.0,
            PipelineStage.BLOCK: 25.0,
            PipelineStage.COMPARE: 50.0,
            PipelineStage.CLASSIFY: 65.0,
            PipelineStage.CLUSTER: 80.0,
            PipelineStage.MERGE: 90.0,
            PipelineStage.COMPLETE: 100.0,
        }
        if self.status == JobStatus.COMPLETED:
            return 100.0
        if self.status in (JobStatus.FAILED, JobStatus.CANCELLED):
            return 0.0
        return stage_progress.get(self.stage, 0.0)

    @property
    def duplicate_rate(self) -> float:
        """Return duplicate rate as a fraction of total records.

        Returns 0.0 if total_records is zero.
        """
        if self.total_records == 0:
            return 0.0
        return self.matched / self.total_records


class PipelineCheckpoint(BaseModel):
    """Checkpoint for resuming a dedup pipeline after interruption.

    Stores the job identifier, current stage, progress counter,
    and serialized state data needed to resume processing from
    the last checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        job_id: Identifier of the dedup job being checkpointed.
        stage: Pipeline stage at the time of checkpoint.
        records_processed: Number of records processed before checkpoint.
        state_data: Serialized pipeline state for resume.
        created_at: Timestamp when the checkpoint was saved.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this checkpoint",
    )
    job_id: str = Field(
        ..., description="Identifier of the dedup job being checkpointed",
    )
    stage: PipelineStage = Field(
        ..., description="Pipeline stage at the time of checkpoint",
    )
    records_processed: int = Field(
        default=0, ge=0,
        description="Number of records processed before checkpoint",
    )
    state_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized pipeline state for resume",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the checkpoint was saved",
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


class DedupStatistics(BaseModel):
    """Aggregated operational statistics for the dedup service.

    Provides high-level metrics for monitoring the overall
    health, throughput, and effectiveness of the deduplication
    pipeline.

    Attributes:
        total_jobs: Total number of dedup jobs executed.
        total_records: Total number of records processed across all jobs.
        total_duplicates: Total number of duplicate pairs found.
        total_merges: Total number of merge operations performed.
        avg_similarity: Average similarity score across all matches.
        duplicate_rate: Overall duplicate rate (duplicates / records).
        avg_cluster_size: Average number of records per duplicate cluster.
        by_status: Count of jobs per status.
        by_strategy: Count of jobs per blocking strategy.
        timestamp: Timestamp when statistics were computed.
    """

    total_jobs: int = Field(
        default=0, ge=0,
        description="Total number of dedup jobs executed",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of records processed across all jobs",
    )
    total_duplicates: int = Field(
        default=0, ge=0,
        description="Total number of duplicate pairs found",
    )
    total_merges: int = Field(
        default=0, ge=0,
        description="Total number of merge operations performed",
    )
    avg_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average similarity score across all matches",
    )
    duplicate_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall duplicate rate (duplicates / records)",
    )
    avg_cluster_size: float = Field(
        default=0.0, ge=0.0,
        description="Average number of records per duplicate cluster",
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of jobs per status",
    )
    by_strategy: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of jobs per blocking strategy",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when statistics were computed",
    )

    model_config = {"extra": "forbid"}


class DedupIssueSummary(BaseModel):
    """Summary of an issue detected during the dedup pipeline.

    Represents a warning, error, or informational message about
    data quality or processing anomalies encountered during
    deduplication.

    Attributes:
        issue_id: Unique identifier for this issue.
        severity: Issue severity classification.
        stage: Pipeline stage where the issue was detected.
        description: Human-readable description of the issue.
        affected_records: Number of records affected by this issue.
        suggested_fix: Recommended remediation action.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    issue_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this issue",
    )
    severity: IssueSeverity = Field(
        default=IssueSeverity.MEDIUM,
        description="Issue severity classification",
    )
    stage: PipelineStage = Field(
        ..., description="Pipeline stage where the issue was detected",
    )
    description: str = Field(
        ..., description="Human-readable description of the issue",
    )
    affected_records: int = Field(
        default=0, ge=0,
        description="Number of records affected by this issue",
    )
    suggested_fix: Optional[str] = Field(
        None,
        description="Recommended remediation action",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is non-empty."""
        if not v or not v.strip():
            raise ValueError("description must be non-empty")
        return v


class DedupJobSummary(BaseModel):
    """Lightweight summary of a dedup job for listing and searching.

    Provides key metadata without loading full cluster and merge
    detail data.

    Attributes:
        job_id: Unique job identifier.
        dataset_ids: List of dataset identifiers processed.
        status: Current job status.
        stage: Current pipeline stage.
        total_records: Total input records.
        matched: Number of matched pairs.
        clustered: Number of clusters formed.
        duplicate_rate: Duplicate rate as a fraction.
        started_at: Job start timestamp.
        completed_at: Job completion timestamp.
    """

    job_id: str = Field(
        ..., description="Unique job identifier",
    )
    dataset_ids: List[str] = Field(
        default_factory=list,
        description="List of dataset identifiers processed",
    )
    status: JobStatus = Field(
        default=JobStatus.COMPLETED,
        description="Current job status",
    )
    stage: PipelineStage = Field(
        default=PipelineStage.COMPLETE,
        description="Current pipeline stage",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total input records",
    )
    matched: int = Field(
        default=0, ge=0,
        description="Number of matched pairs",
    )
    clustered: int = Field(
        default=0, ge=0,
        description="Number of clusters formed",
    )
    duplicate_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Duplicate rate as a fraction",
    )
    started_at: Optional[datetime] = Field(
        None, description="Job start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp",
    )

    model_config = {"extra": "forbid"}

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("job_id must be non-empty")
        return v


class DedupReport(BaseModel):
    """Complete deduplication report for a job.

    Aggregates job summary, cluster details, merge decisions,
    statistics, and issues into a comprehensive report for
    auditing and compliance.

    Attributes:
        report_id: Unique identifier for this report.
        job_id: Identifier of the dedup job this report covers.
        format: Output format of the report.
        summary: High-level text summary of the dedup results.
        clusters: List of duplicate clusters found.
        merge_decisions: List of merge decisions made.
        statistics: Aggregated dedup statistics for this job.
        issues: List of issues encountered during the pipeline.
        generated_at: Timestamp when the report was generated.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    job_id: str = Field(
        ..., description="Identifier of the dedup job this report covers",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format of the report",
    )
    summary: str = Field(
        default="",
        description="High-level text summary of the dedup results",
    )
    clusters: List[DuplicateCluster] = Field(
        default_factory=list,
        description="List of duplicate clusters found",
    )
    merge_decisions: List[MergeDecision] = Field(
        default_factory=list,
        description="List of merge decisions made",
    )
    statistics: Optional[DedupStatistics] = Field(
        None,
        description="Aggregated dedup statistics for this job",
    )
    issues: List[DedupIssueSummary] = Field(
        default_factory=list,
        description="List of issues encountered during the pipeline",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the report was generated",
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


# =============================================================================
# Request Models (7)
# =============================================================================


class FingerprintRequest(BaseModel):
    """Request body for computing record fingerprints.

    Attributes:
        records: List of record dictionaries to fingerprint.
        field_set: List of field names to include in fingerprint.
        algorithm: Fingerprint algorithm to use.
        normalize: Whether to normalize fields before hashing.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to fingerprint",
    )
    field_set: List[str] = Field(
        ..., min_length=1,
        description="List of field names to include in fingerprint",
    )
    algorithm: FingerprintAlgorithm = Field(
        default=FingerprintAlgorithm.SHA256,
        description="Fingerprint algorithm to use",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize fields before hashing",
    )

    model_config = {"extra": "forbid"}

    @field_validator("field_set")
    @classmethod
    def validate_field_set(cls, v: List[str]) -> List[str]:
        """Validate field_set contains no empty strings."""
        for f in v:
            if not f or not f.strip():
                raise ValueError("field_set must not contain empty strings")
        return v


class BlockRequest(BaseModel):
    """Request body for blocking records into candidate groups.

    Attributes:
        fingerprints: List of record fingerprints to block.
        strategy: Blocking strategy to apply.
        key_fields: Field names to use for generating blocking keys.
        window_size: Window size for sorted neighborhood strategy.
    """

    fingerprints: List[RecordFingerprint] = Field(
        ..., min_length=1,
        description="List of record fingerprints to block",
    )
    strategy: BlockingStrategy = Field(
        default=BlockingStrategy.SORTED_NEIGHBORHOOD,
        description="Blocking strategy to apply",
    )
    key_fields: List[str] = Field(
        default_factory=list,
        description="Field names to use for generating blocking keys",
    )
    window_size: int = Field(
        default=10, ge=2, le=1000,
        description="Window size for sorted neighborhood strategy",
    )

    model_config = {"extra": "forbid"}


class CompareRequest(BaseModel):
    """Request body for pairwise record comparison within blocks.

    Attributes:
        block_results: List of block results containing candidate pairs.
        field_configs: List of field comparison configurations.
        sample_rate: Fraction of comparisons to execute (0.0 to 1.0).
    """

    block_results: List[BlockResult] = Field(
        ..., min_length=1,
        description="List of block results containing candidate pairs",
    )
    field_configs: List[FieldComparisonConfig] = Field(
        ..., min_length=1,
        description="List of field comparison configurations",
    )
    sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Fraction of comparisons to execute (0.0 to 1.0)",
    )

    model_config = {"extra": "forbid"}


class ClassifyRequest(BaseModel):
    """Request body for classifying comparison results.

    Attributes:
        comparisons: List of similarity results to classify.
        match_threshold: Score threshold for MATCH classification.
        possible_threshold: Score threshold for POSSIBLE classification.
        use_fellegi_sunter: Whether to use Fellegi-Sunter model.
    """

    comparisons: List[SimilarityResult] = Field(
        ..., min_length=1,
        description="List of similarity results to classify",
    )
    match_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Score threshold for MATCH classification",
    )
    possible_threshold: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="Score threshold for POSSIBLE classification",
    )
    use_fellegi_sunter: bool = Field(
        default=False,
        description="Whether to use Fellegi-Sunter probabilistic model",
    )

    model_config = {"extra": "forbid"}

    @field_validator("match_threshold")
    @classmethod
    def validate_match_threshold(cls, v: float) -> float:
        """Validate match_threshold is reasonable."""
        if v < 0.5:
            raise ValueError(
                "match_threshold must be >= 0.5 to avoid excessive false positives"
            )
        return v


class ClusterRequest(BaseModel):
    """Request body for clustering matched record pairs.

    Attributes:
        matches: List of match results to cluster.
        algorithm: Clustering algorithm to use.
        min_quality: Minimum cluster quality score to accept.
    """

    matches: List[MatchResult] = Field(
        ..., min_length=1,
        description="List of match results to cluster",
    )
    algorithm: ClusterAlgorithm = Field(
        default=ClusterAlgorithm.UNION_FIND,
        description="Clustering algorithm to use",
    )
    min_quality: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Minimum cluster quality score to accept",
    )

    model_config = {"extra": "forbid"}


class MergeRequest(BaseModel):
    """Request body for merging duplicate clusters.

    Attributes:
        clusters: List of duplicate clusters to merge.
        strategy: Merge strategy to apply.
        conflict_resolution: Field-level conflict resolution method.
        source_records: Full source records keyed by record_id for
            field-level merge operations.
    """

    clusters: List[DuplicateCluster] = Field(
        ..., min_length=1,
        description="List of duplicate clusters to merge",
    )
    strategy: MergeStrategy = Field(
        default=MergeStrategy.KEEP_MOST_COMPLETE,
        description="Merge strategy to apply",
    )
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.MOST_COMPLETE,
        description="Field-level conflict resolution method",
    )
    source_records: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Full source records keyed by record_id",
    )

    model_config = {"extra": "forbid"}


class PipelineRequest(BaseModel):
    """Request body for running the full deduplication pipeline.

    Encapsulates input records, the dedup rule to apply, and
    optional pipeline configuration overrides.

    Attributes:
        records: List of record dictionaries to deduplicate.
        rule: Dedup rule defining comparison and merge behavior.
        dataset_ids: List of dataset identifiers for the input records.
        record_source: Whether this is single-dataset or cross-dataset dedup.
        options: Optional pipeline configuration overrides.
    """

    records: List[Dict[str, Any]] = Field(
        ..., min_length=1,
        description="List of record dictionaries to deduplicate",
    )
    rule: DedupRule = Field(
        ..., description="Dedup rule defining comparison and merge behavior",
    )
    dataset_ids: List[str] = Field(
        default_factory=list,
        description="List of dataset identifiers for the input records",
    )
    record_source: RecordSource = Field(
        default=RecordSource.SINGLE_DATASET,
        description="Whether this is single-dataset or cross-dataset dedup",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional pipeline configuration overrides",
    )

    model_config = {"extra": "forbid"}

    @field_validator("records")
    @classmethod
    def validate_records_not_empty(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate records list is non-empty (enforced by min_length)."""
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "DEFAULT_FIELD_WEIGHTS",
    "SIMILARITY_ALGORITHM_DEFAULTS",
    "BLOCKING_STRATEGY_DEFAULTS",
    "MAX_RECORDS_PER_COMPARISON",
    "DEFAULT_NGRAM_SIZE",
    "DEFAULT_MATCH_THRESHOLD",
    "DEFAULT_POSSIBLE_THRESHOLD",
    "DEFAULT_NON_MATCH_THRESHOLD",
    "MAX_BLOCK_COMPARISONS_WARN",
    "FINGERPRINT_ALGORITHMS",
    "BLOCKING_STRATEGIES",
    "SIMILARITY_ALGORITHMS",
    "MERGE_STRATEGIES",
    "CONFLICT_RESOLUTIONS",
    "PIPELINE_STAGE_ORDER",
    "FIELD_TYPES",
    "ISSUE_SEVERITY_ORDER",
    "REPORT_FORMAT_OPTIONS",
    # -------------------------------------------------------------------------
    # Enumerations (13)
    # -------------------------------------------------------------------------
    "FingerprintAlgorithm",
    "BlockingStrategy",
    "SimilarityAlgorithm",
    "MatchClassification",
    "ClusterAlgorithm",
    "MergeStrategy",
    "ConflictResolution",
    "JobStatus",
    "PipelineStage",
    "RecordSource",
    "FieldType",
    "IssueSeverity",
    "ReportFormat",
    # -------------------------------------------------------------------------
    # SDK data models (15)
    # -------------------------------------------------------------------------
    "RecordFingerprint",
    "BlockResult",
    "SimilarityResult",
    "FieldComparisonConfig",
    "MatchResult",
    "MergeConflict",
    "DuplicateCluster",
    "MergeDecision",
    "DedupRule",
    "DedupJob",
    "PipelineCheckpoint",
    "DedupStatistics",
    "DedupIssueSummary",
    "DedupJobSummary",
    "DedupReport",
    # -------------------------------------------------------------------------
    # Request models (7)
    # -------------------------------------------------------------------------
    "FingerprintRequest",
    "BlockRequest",
    "CompareRequest",
    "ClassifyRequest",
    "ClusterRequest",
    "MergeRequest",
    "PipelineRequest",
]
