# -*- coding: utf-8 -*-
"""
Data Lineage Tracker Service Data Models - AGENT-DATA-018

Pydantic v2 data models for the Data Lineage Tracker SDK. Attempts to
re-export Layer 1 enumerations and models from the Data Quality Profiler
(QualityDimension) and Cross-Source Reconciliation (SourceRegistryEngine),
and defines all SDK models for data asset registry, transformation tracking,
lineage graph construction, impact analysis, validation, reporting, change
detection, quality scoring, audit trails, and pipeline orchestration.

Re-exported Layer 1 sources (best-effort, with fallback stubs):
    - greenlang.data_quality_profiler.models: QualityDimension
    - greenlang.cross_source_reconciliation.source_registry: SourceRegistryEngine

New enumerations (14):
    - AssetType, AssetClassification, AssetStatus, TransformationType,
      EdgeType, TraversalDirection, ImpactSeverity, ValidationResult,
      ReportType, ReportFormat, ChangeType, ChangeSeverity, ScoreTier,
      TransformationLogicType

New SDK models (16):
    - DataAsset, TransformationEvent, LineageEdge, GraphSnapshot,
      ImpactAnalysisResult, ValidationReport, LineageReport, ChangeEvent,
      QualityScore, AuditEntry, GraphNode, GraphEdgeView, SubgraphResult,
      LineageChain, LineageStatistics, PipelineResult

Request models (8):
    - RegisterAssetRequest, UpdateAssetRequest, RecordTransformationRequest,
      CreateEdgeRequest, RunImpactAnalysisRequest, RunValidationRequest,
      GenerateReportRequest, RunPipelineRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Layer 1 Re-exports (best-effort with stubs on ImportError)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.models import (  # type: ignore[import]
        QualityDimension as L1QualityDimension,
    )

    QualityDimension = L1QualityDimension
    _DQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DQ_AVAILABLE = False

    class QualityDimension(str, Enum):  # type: ignore[no-redef]
        """Stub re-export when data_quality_profiler is unavailable."""

        COMPLETENESS = "completeness"
        VALIDITY = "validity"
        CONSISTENCY = "consistency"
        TIMELINESS = "timeliness"
        UNIQUENESS = "uniqueness"
        ACCURACY = "accuracy"


try:
    from greenlang.cross_source_reconciliation.source_registry import (  # type: ignore[import]
        SourceRegistryEngine as L1SourceRegistryEngine,
    )

    SourceRegistryEngine = L1SourceRegistryEngine
    _SR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SR_AVAILABLE = False

    class SourceRegistryEngine:  # type: ignore[no-redef]
        """Stub re-export when cross_source_reconciliation is unavailable."""

        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of data assets allowed per tenant / namespace.
MAX_ASSETS_PER_NAMESPACE: int = 50_000

#: Maximum traversal depth for impact analysis and subgraph extraction.
MAX_TRAVERSAL_DEPTH: int = 50

#: Maximum number of edges allowed per asset node.
MAX_EDGES_PER_ASSET: int = 10_000

#: Default confidence score for edges created without explicit confidence.
DEFAULT_EDGE_CONFIDENCE: float = 1.0

#: Minimum confidence threshold for edges to be considered in impact analysis.
MIN_IMPACT_CONFIDENCE: float = 0.1

#: Maximum number of change events stored per graph snapshot pair.
MAX_CHANGE_EVENTS_PER_SNAPSHOT: int = 10_000

#: Default batch size for pipeline operations.
DEFAULT_PIPELINE_BATCH_SIZE: int = 1_000

#: Quality score thresholds for tier classification.
SCORE_TIER_THRESHOLDS: Dict[str, float] = {
    "excellent": 0.90,
    "good": 0.75,
    "fair": 0.50,
    "poor": 0.25,
    "critical": 0.0,
}

#: Supported report formats for lineage visualization and export.
SUPPORTED_REPORT_FORMATS: tuple = ("mermaid", "dot", "json", "d3", "text", "html", "pdf")

#: Impact severity ordering from least to most severe (for comparisons).
IMPACT_SEVERITY_ORDER: tuple = ("low", "medium", "high", "critical")


# =============================================================================
# Enumerations (14)
# =============================================================================


class AssetType(str, Enum):
    """Type classification for a registered data asset.

    Determines how the lineage graph treats the node and which
    metadata fields are expected. Every node in the lineage graph
    must have exactly one AssetType.

    DATASET: A tabular data source (table, file, API response).
    FIELD: A single column or attribute within a dataset.
    AGENT: A GreenLang agent that produces or consumes data.
    PIPELINE: An orchestrated workflow of multiple agents.
    REPORT: A generated output report (CSRD, GHG, SOC 2).
    METRIC: A computed metric or KPI derived from data.
    EXTERNAL_SOURCE: An external system or data provider.
    """

    DATASET = "dataset"
    FIELD = "field"
    AGENT = "agent"
    PIPELINE = "pipeline"
    REPORT = "report"
    METRIC = "metric"
    EXTERNAL_SOURCE = "external_source"


class AssetClassification(str, Enum):
    """Data classification level for access control and governance.

    Determines visibility, encryption, and audit requirements
    for a data asset. Higher classifications require stricter
    controls and more detailed audit trails.

    PUBLIC: No access restrictions; safe for external sharing.
    INTERNAL: Accessible to all internal users and agents.
    CONFIDENTIAL: Restricted to authorized roles and teams.
    RESTRICTED: Highest sensitivity; requires explicit approval.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class AssetStatus(str, Enum):
    """Lifecycle status of a registered data asset.

    Controls whether downstream consumers can reference the asset
    and whether lineage edges can be created to or from it.

    ACTIVE: Asset is in production use and fully operational.
    DEPRECATED: Asset is superseded; consumers should migrate.
    ARCHIVED: Asset is retired and no longer usable.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TransformationType(str, Enum):
    """Type of data transformation applied by an agent or pipeline step.

    Classifies the nature of data processing performed between
    source and target assets. Used for transformation cataloging,
    impact analysis, and lineage visualization.

    FILTER: Rows are removed based on a predicate.
    AGGREGATE: Rows are grouped and summarized (sum, avg, count).
    JOIN: Two or more datasets are combined on a key.
    CALCULATE: New values are derived from existing fields.
    IMPUTE: Missing values are filled using statistical methods.
    DEDUPLICATE: Duplicate records are detected and merged.
    ENRICH: External data is added to existing records.
    MERGE: Multiple datasets are unioned or concatenated.
    SPLIT: A single dataset is partitioned into multiple outputs.
    VALIDATE: Data is checked against quality rules.
    NORMALIZE: Values are standardized (units, formats, encodings).
    CLASSIFY: Records are categorized or labeled.
    """

    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    CALCULATE = "calculate"
    IMPUTE = "impute"
    DEDUPLICATE = "deduplicate"
    ENRICH = "enrich"
    MERGE = "merge"
    SPLIT = "split"
    VALIDATE = "validate"
    NORMALIZE = "normalize"
    CLASSIFY = "classify"


class EdgeType(str, Enum):
    """Granularity level of a lineage edge in the graph.

    Determines whether the edge represents a relationship between
    entire datasets or between individual columns/fields within
    datasets. Column-level lineage provides finer-grained tracing.

    DATASET_LEVEL: Edge connects two dataset-level assets.
    COLUMN_LEVEL: Edge connects individual fields across datasets.
    """

    DATASET_LEVEL = "dataset_level"
    COLUMN_LEVEL = "column_level"


class TraversalDirection(str, Enum):
    """Direction of graph traversal for impact analysis and lineage queries.

    FORWARD: Traverse from source to downstream consumers (impact).
    BACKWARD: Traverse from target to upstream producers (provenance).
    """

    FORWARD = "forward"
    BACKWARD = "backward"


class ImpactSeverity(str, Enum):
    """Severity classification for an affected asset in impact analysis.

    Quantifies the degree to which a change to a root asset
    would affect a downstream or upstream dependent asset.

    CRITICAL: Immediate breakage; data loss or compliance risk.
    HIGH: Likely pipeline failure or significant data quality degradation.
    MEDIUM: Potential warnings or soft failures in downstream processing.
    LOW: Cosmetic or informational impact only.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ValidationResult(str, Enum):
    """Outcome of a lineage graph validation check.

    PASS_RESULT: All validation checks passed with no issues.
    WARN: Validation completed with non-blocking warnings.
    FAIL: Validation detected blocking issues that must be resolved.

    Note:
        The value "pass" is used for serialization; the Python member
        name is PASS_RESULT to avoid conflict with the reserved keyword.
    """

    PASS_RESULT = "pass"
    WARN = "warn"
    FAIL = "fail"


class ReportType(str, Enum):
    """Type of lineage report to generate.

    Determines the structure, content, and compliance framework
    alignment of the generated lineage report.

    CSRD_ESRS: EU Corporate Sustainability Reporting Directive lineage.
    GHG_PROTOCOL: GHG Protocol Scope 1/2/3 data provenance report.
    SOC2: SOC 2 Type II data lineage audit evidence.
    CUSTOM: User-defined report structure and content.
    VISUALIZATION: Interactive graph visualization export.
    """

    CSRD_ESRS = "csrd_esrs"
    GHG_PROTOCOL = "ghg_protocol"
    SOC2 = "soc2"
    CUSTOM = "custom"
    VISUALIZATION = "visualization"


class ReportFormat(str, Enum):
    """Output format for a lineage report or visualization.

    MERMAID: Mermaid diagram syntax (Markdown-embeddable).
    DOT: Graphviz DOT language for graph rendering.
    JSON: Structured JSON for programmatic consumption.
    D3: D3.js-compatible JSON for interactive web visualization.
    TEXT: Plain-text summary for terminal or log output.
    HTML: Self-contained HTML page with embedded visualization.
    PDF: Portable Document Format for formal distribution.
    """

    MERMAID = "mermaid"
    DOT = "dot"
    JSON = "json"
    D3 = "d3"
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"


class ChangeType(str, Enum):
    """Type of structural change detected between two graph snapshots.

    Used to classify each diff entry produced by the change detection
    engine when comparing successive snapshots of the lineage graph.

    NODE_ADDED: A new asset node was added to the graph.
    NODE_REMOVED: An existing asset node was removed from the graph.
    EDGE_ADDED: A new lineage edge was added between nodes.
    EDGE_REMOVED: An existing lineage edge was removed.
    TOPOLOGY_CHANGED: The overall graph topology changed significantly.
    """

    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"
    TOPOLOGY_CHANGED = "topology_changed"


class ChangeSeverity(str, Enum):
    """Severity classification for a graph change event.

    LOW: Change is cosmetic or affects only metadata.
    MEDIUM: Change may affect downstream consumers.
    HIGH: Change is likely to cause pipeline or data quality issues.
    CRITICAL: Change causes immediate breakage or compliance risk.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScoreTier(str, Enum):
    """Quality score tier for data lineage completeness and health.

    Used to bucket a numeric quality score into a human-readable
    tier for dashboards, alerts, and governance reporting.

    EXCELLENT: Score >= 0.90; lineage is comprehensive and current.
    GOOD: Score >= 0.75; lineage covers most critical paths.
    FAIR: Score >= 0.50; lineage has notable gaps.
    POOR: Score >= 0.25; lineage is significantly incomplete.
    CRITICAL: Score < 0.25; lineage is largely missing or stale.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class TransformationLogicType(str, Enum):
    """Type of column-level transformation logic recorded on a lineage edge.

    Describes how a specific source field is transformed into a
    target field. Stored on column-level edges to enable precise
    provenance tracing and impact analysis at the field level.

    RENAME: Source field is renamed without value transformation.
    CAST: Source field value is cast to a different data type.
    AGGREGATE_FUNC: Source field is aggregated (SUM, AVG, COUNT, etc.).
    COMPUTE: Target field is derived from an expression over source fields.
    CONDITIONAL: Target field is set based on conditional logic (CASE/IF).
    LOOKUP: Target field is resolved from a lookup table or reference data.
    MERGE_FIELDS: Multiple source fields are concatenated or combined.
    SPLIT_FIELD: A single source field is split into multiple target fields.
    """

    RENAME = "rename"
    CAST = "cast"
    AGGREGATE_FUNC = "aggregate_func"
    COMPUTE = "compute"
    CONDITIONAL = "conditional"
    LOOKUP = "lookup"
    MERGE_FIELDS = "merge_fields"
    SPLIT_FIELD = "split_field"


# =============================================================================
# SDK Data Models (16)
# =============================================================================


class DataAsset(BaseModel):
    """A registered data asset in the Data Lineage Tracker registry.

    Represents any addressable data entity in the GreenLang platform:
    datasets, fields, agents, pipelines, reports, metrics, or external
    sources. Each asset is a node in the lineage graph and carries
    metadata for classification, governance, and discovery.

    Attributes:
        id: Unique asset identifier (UUID v4).
        qualified_name: Fully qualified, globally unique name
            (e.g., "erp.sap.spend_data.amount_usd").
        asset_type: Classification of the asset's role in the data flow.
        display_name: Human-readable display name for UIs and reports.
        owner: Team or service responsible for this asset.
        tags: Arbitrary key-value labels for discovery and filtering.
        classification: Data sensitivity classification level.
        status: Current lifecycle status of the asset.
        schema_ref: Optional reference to a schema definition ID.
        description: Human-readable description of the asset's purpose.
        metadata: Additional unstructured metadata key-value pairs.
        created_at: UTC timestamp when the asset was first registered.
        updated_at: UTC timestamp when the asset was last updated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique asset identifier (UUID v4)",
    )
    qualified_name: str = Field(
        ...,
        description="Fully qualified, globally unique asset name",
    )
    asset_type: AssetType = Field(
        ...,
        description="Classification of the asset's role in the data flow",
    )
    display_name: str = Field(
        default="",
        description="Human-readable display name for UIs and reports",
    )
    owner: str = Field(
        default="",
        description="Team or service responsible for this asset",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for discovery and filtering",
    )
    classification: AssetClassification = Field(
        default=AssetClassification.INTERNAL,
        description="Data sensitivity classification level",
    )
    status: AssetStatus = Field(
        default=AssetStatus.ACTIVE,
        description="Current lifecycle status of the asset",
    )
    schema_ref: Optional[str] = Field(
        None,
        description="Optional reference to a schema definition ID",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the asset's purpose",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional unstructured metadata key-value pairs",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the asset was first registered",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the asset was last updated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("qualified_name")
    @classmethod
    def validate_qualified_name(cls, v: str) -> str:
        """Validate qualified_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("qualified_name must be non-empty")
        return v


class TransformationEvent(BaseModel):
    """A recorded data transformation event in the lineage graph.

    Captures the details of a single transformation operation performed
    by an agent or pipeline step, including record-level metrics for
    data quality monitoring and audit trail construction.

    Attributes:
        id: Unique transformation event identifier (UUID v4).
        transformation_type: Type of transformation performed.
        agent_id: ID of the agent that performed the transformation.
        pipeline_id: ID of the pipeline containing this transformation.
        execution_id: Unique identifier for this execution run.
        source_assets: List of source asset IDs consumed by the transform.
        target_assets: List of target asset IDs produced by the transform.
        description: Human-readable description of what the transform does.
        parameters: Transformation-specific parameters and configuration.
        records_in: Number of input records consumed.
        records_out: Number of output records produced.
        records_filtered: Number of records removed by filtering.
        records_error: Number of records that failed transformation.
        duration_ms: Wall-clock duration of the transformation in milliseconds.
        started_at: UTC timestamp when the transformation began.
        completed_at: UTC timestamp when the transformation ended.
        metadata: Additional unstructured metadata key-value pairs.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transformation event identifier (UUID v4)",
    )
    transformation_type: TransformationType = Field(
        ...,
        description="Type of transformation performed",
    )
    agent_id: str = Field(
        default="",
        description="ID of the agent that performed the transformation",
    )
    pipeline_id: str = Field(
        default="",
        description="ID of the pipeline containing this transformation",
    )
    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this execution run",
    )
    source_assets: List[str] = Field(
        default_factory=list,
        description="List of source asset IDs consumed by the transform",
    )
    target_assets: List[str] = Field(
        default_factory=list,
        description="List of target asset IDs produced by the transform",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the transformation",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transformation-specific parameters and configuration",
    )
    records_in: int = Field(
        default=0,
        ge=0,
        description="Number of input records consumed",
    )
    records_out: int = Field(
        default=0,
        ge=0,
        description="Number of output records produced",
    )
    records_filtered: int = Field(
        default=0,
        ge=0,
        description="Number of records removed by filtering",
    )
    records_error: int = Field(
        default=0,
        ge=0,
        description="Number of records that failed transformation",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration in milliseconds",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the transformation began",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the transformation ended",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional unstructured metadata key-value pairs",
    )

    model_config = {"extra": "forbid"}

    @field_validator("transformation_type")
    @classmethod
    def validate_transformation_type(cls, v: TransformationType) -> TransformationType:
        """Validate transformation_type is a valid enum member."""
        if not isinstance(v, TransformationType):
            raise ValueError(f"Invalid transformation_type: {v}")
        return v


class LineageEdge(BaseModel):
    """A directed edge in the lineage graph connecting two data assets.

    Represents a data flow relationship from a source asset to a target
    asset, optionally through a recorded transformation. Edges can be
    dataset-level or column-level for fine-grained lineage tracing.

    Attributes:
        id: Unique edge identifier (UUID v4).
        source_asset_id: ID of the upstream source asset.
        target_asset_id: ID of the downstream target asset.
        transformation_id: Optional reference to a TransformationEvent ID.
        edge_type: Granularity level (dataset or column).
        source_field: Source field name (for column-level edges).
        target_field: Target field name (for column-level edges).
        transformation_logic: Description of the column-level transform.
        confidence: Confidence score for this edge (0.0 to 1.0).
        created_at: UTC timestamp when the edge was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique edge identifier (UUID v4)",
    )
    source_asset_id: str = Field(
        ...,
        description="ID of the upstream source asset",
    )
    target_asset_id: str = Field(
        ...,
        description="ID of the downstream target asset",
    )
    transformation_id: Optional[str] = Field(
        None,
        description="Optional reference to a TransformationEvent ID",
    )
    edge_type: EdgeType = Field(
        default=EdgeType.DATASET_LEVEL,
        description="Granularity level (dataset or column)",
    )
    source_field: Optional[str] = Field(
        None,
        description="Source field name (for column-level edges only)",
    )
    target_field: Optional[str] = Field(
        None,
        description="Target field name (for column-level edges only)",
    )
    transformation_logic: Optional[str] = Field(
        None,
        description="Description of the column-level transformation logic",
    )
    confidence: float = Field(
        default=DEFAULT_EDGE_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Confidence score for this edge (0.0 to 1.0)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the edge was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_asset_id")
    @classmethod
    def validate_source_asset_id(cls, v: str) -> str:
        """Validate source_asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_asset_id must be non-empty")
        return v

    @field_validator("target_asset_id")
    @classmethod
    def validate_target_asset_id(cls, v: str) -> str:
        """Validate target_asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_asset_id must be non-empty")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class GraphSnapshot(BaseModel):
    """A point-in-time snapshot of the lineage graph topology.

    Snapshots are periodically captured to enable change detection,
    drift monitoring, and historical comparison of the lineage graph.
    Each snapshot records aggregate metrics about the graph structure
    and a SHA-256 hash for tamper detection.

    Attributes:
        id: Unique snapshot identifier (UUID v4).
        snapshot_name: Human-readable name for this snapshot.
        node_count: Total number of asset nodes in the graph.
        edge_count: Total number of lineage edges in the graph.
        max_depth: Maximum depth of the longest lineage chain.
        connected_components: Number of disconnected subgraphs.
        orphan_count: Number of nodes with no incoming or outgoing edges.
        coverage_score: Lineage coverage as a fraction (0.0 to 1.0).
        graph_hash: SHA-256 hash of the graph topology for tamper detection.
        created_at: UTC timestamp when the snapshot was captured.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique snapshot identifier (UUID v4)",
    )
    snapshot_name: str = Field(
        default="",
        description="Human-readable name for this snapshot",
    )
    node_count: int = Field(
        default=0,
        ge=0,
        description="Total number of asset nodes in the graph",
    )
    edge_count: int = Field(
        default=0,
        ge=0,
        description="Total number of lineage edges in the graph",
    )
    max_depth: int = Field(
        default=0,
        ge=0,
        description="Maximum depth of the longest lineage chain",
    )
    connected_components: int = Field(
        default=0,
        ge=0,
        description="Number of disconnected subgraphs",
    )
    orphan_count: int = Field(
        default=0,
        ge=0,
        description="Number of nodes with no incoming or outgoing edges",
    )
    coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Lineage coverage as a fraction (0.0 to 1.0)",
    )
    graph_hash: str = Field(
        default="",
        description="SHA-256 hash of the graph topology for tamper detection",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the snapshot was captured",
    )

    model_config = {"extra": "forbid"}

    @field_validator("coverage_score")
    @classmethod
    def validate_coverage_score(cls, v: float) -> float:
        """Validate coverage_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"coverage_score must be between 0.0 and 1.0, got {v}")
        return v


class ImpactAnalysisResult(BaseModel):
    """Result of an impact analysis traversal from a root asset.

    Produced by traversing the lineage graph forward (downstream impact)
    or backward (upstream provenance) from a root asset to a specified
    depth. Identifies all affected assets with severity classifications
    and calculates the blast radius.

    Attributes:
        id: Unique result identifier (UUID v4).
        root_asset_id: ID of the asset from which traversal started.
        direction: Direction of graph traversal (forward or backward).
        depth: Maximum traversal depth reached.
        affected_assets: List of affected asset details (id, name, severity, path).
        affected_assets_count: Total number of affected assets found.
        critical_count: Number of assets with CRITICAL severity.
        high_count: Number of assets with HIGH severity.
        medium_count: Number of assets with MEDIUM severity.
        low_count: Number of assets with LOW severity.
        blast_radius: Fraction of total graph nodes affected (0.0 to 1.0).
        created_at: UTC timestamp when the analysis was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier (UUID v4)",
    )
    root_asset_id: str = Field(
        ...,
        description="ID of the asset from which traversal started",
    )
    direction: TraversalDirection = Field(
        default=TraversalDirection.FORWARD,
        description="Direction of graph traversal (forward or backward)",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Maximum traversal depth reached",
    )
    affected_assets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of affected asset details (id, name, severity, path)",
    )
    affected_assets_count: int = Field(
        default=0,
        ge=0,
        description="Total number of affected assets found",
    )
    critical_count: int = Field(
        default=0,
        ge=0,
        description="Number of assets with CRITICAL severity",
    )
    high_count: int = Field(
        default=0,
        ge=0,
        description="Number of assets with HIGH severity",
    )
    medium_count: int = Field(
        default=0,
        ge=0,
        description="Number of assets with MEDIUM severity",
    )
    low_count: int = Field(
        default=0,
        ge=0,
        description="Number of assets with LOW severity",
    )
    blast_radius: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of total graph nodes affected (0.0 to 1.0)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the analysis was performed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("root_asset_id")
    @classmethod
    def validate_root_asset_id(cls, v: str) -> str:
        """Validate root_asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("root_asset_id must be non-empty")
        return v

    @field_validator("blast_radius")
    @classmethod
    def validate_blast_radius(cls, v: float) -> float:
        """Validate blast_radius is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"blast_radius must be between 0.0 and 1.0, got {v}")
        return v


class ValidationReport(BaseModel):
    """Result of a lineage graph validation and health check.

    Produced by the validation engine to detect structural issues
    in the lineage graph such as orphan nodes (no edges), broken
    edges (dangling references), cycles, and coverage gaps.

    Attributes:
        id: Unique report identifier (UUID v4).
        scope: Scope of the validation (e.g., "full", "namespace:xyz").
        orphan_nodes: Number of nodes with no incoming or outgoing edges.
        broken_edges: Number of edges referencing non-existent nodes.
        cycles_detected: Number of cycles found in the directed graph.
        source_coverage: Fraction of sources with complete lineage (0.0-1.0).
        completeness_score: Overall lineage completeness score (0.0-1.0).
        freshness_score: Lineage freshness score based on update recency (0.0-1.0).
        issues: List of specific validation issue details.
        recommendations: List of suggested remediation actions.
        validated_at: UTC timestamp when the validation was performed.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier (UUID v4)",
    )
    scope: str = Field(
        default="full",
        description="Scope of the validation (e.g., 'full', 'namespace:xyz')",
    )
    orphan_nodes: int = Field(
        default=0,
        ge=0,
        description="Number of nodes with no incoming or outgoing edges",
    )
    broken_edges: int = Field(
        default=0,
        ge=0,
        description="Number of edges referencing non-existent nodes",
    )
    cycles_detected: int = Field(
        default=0,
        ge=0,
        description="Number of cycles found in the directed graph",
    )
    source_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of sources with complete lineage (0.0 to 1.0)",
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall lineage completeness score (0.0 to 1.0)",
    )
    freshness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Lineage freshness score based on update recency (0.0 to 1.0)",
    )
    issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of specific validation issue details",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of suggested remediation actions",
    )
    validated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the validation was performed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_coverage")
    @classmethod
    def validate_source_coverage(cls, v: float) -> float:
        """Validate source_coverage is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"source_coverage must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("completeness_score")
    @classmethod
    def validate_completeness_score(cls, v: float) -> float:
        """Validate completeness_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"completeness_score must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("freshness_score")
    @classmethod
    def validate_freshness_score(cls, v: float) -> float:
        """Validate freshness_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"freshness_score must be between 0.0 and 1.0, got {v}")
        return v


class LineageReport(BaseModel):
    """A generated lineage report in a specified format.

    Produced by the reporting engine to render lineage information
    for compliance evidence, data governance documentation, or
    interactive visualization. Reports are immutable once generated
    and include a SHA-256 hash for tamper detection.

    Attributes:
        id: Unique report identifier (UUID v4).
        report_type: Type of lineage report (CSRD, GHG, SOC2, etc.).
        format: Output format (mermaid, dot, json, d3, text, html, pdf).
        scope: Scope of the report (e.g., "full", "asset:xyz").
        parameters: Report generation parameters and configuration.
        content: The rendered report content as a string.
        report_hash: SHA-256 hash of the report content for tamper detection.
        generated_by: Actor (user or service) that requested the report.
        generated_at: UTC timestamp when the report was generated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier (UUID v4)",
    )
    report_type: ReportType = Field(
        default=ReportType.CUSTOM,
        description="Type of lineage report (CSRD, GHG, SOC2, etc.)",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format (mermaid, dot, json, d3, text, html, pdf)",
    )
    scope: str = Field(
        default="full",
        description="Scope of the report (e.g., 'full', 'asset:xyz')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report generation parameters and configuration",
    )
    content: str = Field(
        default="",
        description="The rendered report content as a string",
    )
    report_hash: str = Field(
        default="",
        description="SHA-256 hash of the report content for tamper detection",
    )
    generated_by: str = Field(
        default="system",
        description="Actor (user or service) that requested the report",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the report was generated",
    )

    model_config = {"extra": "forbid"}


class ChangeEvent(BaseModel):
    """A structural change detected between two lineage graph snapshots.

    Produced by the change detection engine when comparing consecutive
    snapshots. Each event represents a single node or edge addition,
    removal, or topology change with its associated severity.

    Attributes:
        id: Unique change event identifier (UUID v4).
        previous_snapshot_id: ID of the earlier graph snapshot.
        current_snapshot_id: ID of the later graph snapshot.
        change_type: Category of structural change detected.
        entity_id: ID of the affected node or edge.
        entity_type: Type of the affected entity (e.g., "DataAsset", "LineageEdge").
        details: Structured details about the change (before/after state).
        severity: Impact severity of this change.
        detected_at: UTC timestamp when the change was detected.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique change event identifier (UUID v4)",
    )
    previous_snapshot_id: str = Field(
        ...,
        description="ID of the earlier graph snapshot",
    )
    current_snapshot_id: str = Field(
        ...,
        description="ID of the later graph snapshot",
    )
    change_type: ChangeType = Field(
        ...,
        description="Category of structural change detected",
    )
    entity_id: str = Field(
        ...,
        description="ID of the affected node or edge",
    )
    entity_type: str = Field(
        default="",
        description="Type of the affected entity (e.g., 'DataAsset', 'LineageEdge')",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the change (before/after state)",
    )
    severity: ChangeSeverity = Field(
        default=ChangeSeverity.LOW,
        description="Impact severity of this change",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the change was detected",
    )

    model_config = {"extra": "forbid"}

    @field_validator("previous_snapshot_id")
    @classmethod
    def validate_previous_snapshot_id(cls, v: str) -> str:
        """Validate previous_snapshot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("previous_snapshot_id must be non-empty")
        return v

    @field_validator("current_snapshot_id")
    @classmethod
    def validate_current_snapshot_id(cls, v: str) -> str:
        """Validate current_snapshot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("current_snapshot_id must be non-empty")
        return v

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v


class QualityScore(BaseModel):
    """A composite quality score for a data asset's lineage health.

    Aggregates multiple quality dimensions into a single overall score
    for governance dashboards and alerting. Factors include source
    credibility, transformation depth, freshness, documentation
    completeness, and manual intervention frequency.

    Attributes:
        id: Unique score identifier (UUID v4).
        asset_id: ID of the data asset being scored.
        source_credibility: Credibility score of upstream sources (0.0-1.0).
        transformation_depth: Penalty factor for deep transformation chains.
        freshness_score: Freshness of lineage metadata (0.0-1.0).
        documentation_score: Completeness of asset documentation (0.0-1.0).
        manual_intervention_count: Number of manual edits to lineage.
        overall_score: Weighted composite quality score (0.0-1.0).
        scoring_details: Detailed breakdown of score components.
        scored_at: UTC timestamp when the score was calculated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique score identifier (UUID v4)",
    )
    asset_id: str = Field(
        ...,
        description="ID of the data asset being scored",
    )
    source_credibility: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Credibility score of upstream sources (0.0 to 1.0)",
    )
    transformation_depth: int = Field(
        default=0,
        ge=0,
        description="Number of transformation hops from original source",
    )
    freshness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Freshness of lineage metadata (0.0 to 1.0)",
    )
    documentation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completeness of asset documentation (0.0 to 1.0)",
    )
    manual_intervention_count: int = Field(
        default=0,
        ge=0,
        description="Number of manual edits to lineage for this asset",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted composite quality score (0.0 to 1.0)",
    )
    scoring_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed breakdown of score components and weights",
    )
    scored_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the score was calculated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("asset_id must be non-empty")
        return v

    @field_validator("source_credibility")
    @classmethod
    def validate_source_credibility(cls, v: float) -> float:
        """Validate source_credibility is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"source_credibility must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        """Validate overall_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"overall_score must be between 0.0 and 1.0, got {v}")
        return v


class AuditEntry(BaseModel):
    """An immutable audit log entry for any data lineage action.

    All create, update, delete, and query actions in the Data Lineage
    Tracker produce an AuditEntry. Entries form a provenance chain
    using SHA-256 hashes linking each entry to its parent for
    tamper-evident audit trails.

    Attributes:
        id: Unique audit entry identifier (UUID v4).
        action: Action verb (e.g., "register_asset", "create_edge").
        entity_type: Type of entity acted upon (e.g., "DataAsset").
        entity_id: ID of the entity that was acted upon.
        actor: User, service, or system that performed the action.
        details: Structured details about the action and its parameters.
        previous_state: Snapshot of the entity state before the action.
        new_state: Snapshot of the entity state after the action.
        provenance_hash: SHA-256 hash of this entry's content.
        parent_hash: SHA-256 hash of the immediately preceding audit entry.
        created_at: UTC timestamp when the audit entry was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique audit entry identifier (UUID v4)",
    )
    action: str = Field(
        ...,
        description="Action verb (e.g., 'register_asset', 'create_edge')",
    )
    entity_type: str = Field(
        ...,
        description="Type of entity acted upon (e.g., 'DataAsset')",
    )
    entity_id: str = Field(
        ...,
        description="ID of the entity that was acted upon",
    )
    actor: str = Field(
        default="system",
        description="User, service, or system that performed the action",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the action and its parameters",
    )
    previous_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot of the entity state before the action",
    )
    new_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot of the entity state after the action",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of this entry's content for tamper detection",
    )
    parent_hash: str = Field(
        default="",
        description="SHA-256 hash of the immediately preceding audit entry",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the audit entry was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is non-empty."""
        if not v or not v.strip():
            raise ValueError("action must be non-empty")
        return v

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate entity_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_type must be non-empty")
        return v

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v


class GraphNode(BaseModel):
    """A lightweight view of an asset node in the lineage graph.

    Used in subgraph query results and visualization payloads to
    provide essential node metadata without the full DataAsset weight.

    Attributes:
        asset_id: ID of the data asset this node represents.
        qualified_name: Fully qualified asset name.
        asset_type: Type classification of the asset.
        metadata: Subset of asset metadata relevant to the query.
        upstream_count: Number of incoming edges (upstream dependencies).
        downstream_count: Number of outgoing edges (downstream consumers).
    """

    asset_id: str = Field(
        ...,
        description="ID of the data asset this node represents",
    )
    qualified_name: str = Field(
        default="",
        description="Fully qualified asset name",
    )
    asset_type: AssetType = Field(
        default=AssetType.DATASET,
        description="Type classification of the asset",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Subset of asset metadata relevant to the query",
    )
    upstream_count: int = Field(
        default=0,
        ge=0,
        description="Number of incoming edges (upstream dependencies)",
    )
    downstream_count: int = Field(
        default=0,
        ge=0,
        description="Number of outgoing edges (downstream consumers)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("asset_id must be non-empty")
        return v


class GraphEdgeView(BaseModel):
    """A lightweight view of a lineage edge for graph query results.

    Used in subgraph query results and visualization payloads to
    provide essential edge metadata without the full LineageEdge weight.

    Attributes:
        edge_id: ID of the lineage edge.
        source_id: ID of the upstream source asset.
        target_id: ID of the downstream target asset.
        transformation_type: Type of transformation on this edge.
        edge_type: Granularity level (dataset or column).
        confidence: Confidence score for this edge (0.0 to 1.0).
    """

    edge_id: str = Field(
        ...,
        description="ID of the lineage edge",
    )
    source_id: str = Field(
        ...,
        description="ID of the upstream source asset",
    )
    target_id: str = Field(
        ...,
        description="ID of the downstream target asset",
    )
    transformation_type: Optional[str] = Field(
        None,
        description="Type of transformation on this edge",
    )
    edge_type: EdgeType = Field(
        default=EdgeType.DATASET_LEVEL,
        description="Granularity level (dataset or column)",
    )
    confidence: float = Field(
        default=DEFAULT_EDGE_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Confidence score for this edge (0.0 to 1.0)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("edge_id")
    @classmethod
    def validate_edge_id(cls, v: str) -> str:
        """Validate edge_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("edge_id must be non-empty")
        return v

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, v: str) -> str:
        """Validate target_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_id must be non-empty")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class SubgraphResult(BaseModel):
    """Result of a subgraph extraction query around a root asset.

    Contains the set of nodes and edges reachable from the root
    asset within the specified traversal depth. Used for focused
    lineage visualization and scoped impact analysis.

    Attributes:
        root_asset_id: ID of the asset at the center of the subgraph.
        depth: Maximum traversal depth from the root.
        nodes: List of graph nodes in the subgraph.
        edges: List of graph edges in the subgraph.
        node_count: Total number of nodes in the subgraph.
        edge_count: Total number of edges in the subgraph.
    """

    root_asset_id: str = Field(
        ...,
        description="ID of the asset at the center of the subgraph",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Maximum traversal depth from the root",
    )
    nodes: List[GraphNode] = Field(
        default_factory=list,
        description="List of graph nodes in the subgraph",
    )
    edges: List[GraphEdgeView] = Field(
        default_factory=list,
        description="List of graph edges in the subgraph",
    )
    node_count: int = Field(
        default=0,
        ge=0,
        description="Total number of nodes in the subgraph",
    )
    edge_count: int = Field(
        default=0,
        ge=0,
        description="Total number of edges in the subgraph",
    )

    model_config = {"extra": "forbid"}

    @field_validator("root_asset_id")
    @classmethod
    def validate_root_asset_id(cls, v: str) -> str:
        """Validate root_asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("root_asset_id must be non-empty")
        return v


class LineageChain(BaseModel):
    """An ordered chain of lineage steps from a root asset.

    Represents a single path through the lineage graph, either
    forward (source to final consumer) or backward (target to
    original source). Each step in the chain includes the asset,
    the transformation applied, and the edge traversed.

    Attributes:
        asset_id: ID of the root asset for this chain.
        direction: Direction of the chain traversal.
        chain: Ordered list of chain step details
            (asset_id, qualified_name, transformation, depth).
        depth: Total depth of the chain.
        total_transformations: Number of transformations in the chain.
    """

    asset_id: str = Field(
        ...,
        description="ID of the root asset for this chain",
    )
    direction: TraversalDirection = Field(
        default=TraversalDirection.BACKWARD,
        description="Direction of the chain traversal",
    )
    chain: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of chain step details",
    )
    depth: int = Field(
        default=0,
        ge=0,
        description="Total depth of the chain",
    )
    total_transformations: int = Field(
        default=0,
        ge=0,
        description="Number of transformations in the chain",
    )

    model_config = {"extra": "forbid"}

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("asset_id must be non-empty")
        return v


class LineageStatistics(BaseModel):
    """Aggregated operational statistics for the Data Lineage Tracker service.

    Provides high-level metrics for service health monitoring, capacity
    planning, and SLO tracking. All counts reflect the current state
    of the lineage graph.

    Attributes:
        total_assets: Total number of registered data assets.
        total_transformations: Total number of recorded transformation events.
        total_edges: Total number of lineage edges in the graph.
        avg_depth: Average lineage chain depth across all terminal assets.
        max_depth: Maximum lineage chain depth in the graph.
        coverage_score: Overall lineage coverage (0.0 to 1.0).
        orphan_count: Number of assets with no lineage connections.
        assets_by_type: Asset count broken down by AssetType.
        transformations_by_type: Transformation count broken down by TransformationType.
    """

    total_assets: int = Field(
        default=0,
        ge=0,
        description="Total number of registered data assets",
    )
    total_transformations: int = Field(
        default=0,
        ge=0,
        description="Total number of recorded transformation events",
    )
    total_edges: int = Field(
        default=0,
        ge=0,
        description="Total number of lineage edges in the graph",
    )
    avg_depth: float = Field(
        default=0.0,
        ge=0.0,
        description="Average lineage chain depth across all terminal assets",
    )
    max_depth: int = Field(
        default=0,
        ge=0,
        description="Maximum lineage chain depth in the graph",
    )
    coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall lineage coverage (0.0 to 1.0)",
    )
    orphan_count: int = Field(
        default=0,
        ge=0,
        description="Number of assets with no lineage connections",
    )
    assets_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Asset count broken down by AssetType",
    )
    transformations_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Transformation count broken down by TransformationType",
    )

    model_config = {"extra": "forbid"}

    @field_validator("coverage_score")
    @classmethod
    def validate_coverage_score(cls, v: float) -> float:
        """Validate coverage_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"coverage_score must be between 0.0 and 1.0, got {v}")
        return v


class PipelineResult(BaseModel):
    """The complete result of a full data lineage pipeline run.

    A pipeline run encompasses all stages: asset registration,
    transformation capture, edge creation, validation, and report
    generation. The PipelineResult aggregates the outputs of each
    stage into a single return value from the RunPipeline endpoint.

    Attributes:
        pipeline_id: Unique identifier for this pipeline run (UUID v4).
        stages_completed: Number of pipeline stages that completed.
        assets_registered: Number of data assets registered during the run.
        transformations_captured: Number of transformations recorded.
        edges_created: Number of lineage edges created.
        validation_result: Outcome of the validation stage (pass/warn/fail).
        report_generated: Whether a lineage report was generated.
        duration_ms: Total wall-clock time for the pipeline run in milliseconds.
        errors: List of error messages from failed stages.
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this pipeline run (UUID v4)",
    )
    stages_completed: int = Field(
        default=0,
        ge=0,
        description="Number of pipeline stages that completed successfully",
    )
    assets_registered: int = Field(
        default=0,
        ge=0,
        description="Number of data assets registered during the run",
    )
    transformations_captured: int = Field(
        default=0,
        ge=0,
        description="Number of transformations recorded during the run",
    )
    edges_created: int = Field(
        default=0,
        ge=0,
        description="Number of lineage edges created during the run",
    )
    validation_result: Optional[str] = Field(
        None,
        description="Outcome of the validation stage (pass/warn/fail or None)",
    )
    report_generated: bool = Field(
        default=False,
        description="Whether a lineage report was generated",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock time in milliseconds",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages from failed stages",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models (8)
# =============================================================================


class RegisterAssetRequest(BaseModel):
    """Request body for registering a new data asset.

    Attributes:
        qualified_name: Fully qualified, globally unique asset name.
        asset_type: Classification of the asset's role in the data flow.
        display_name: Human-readable display name.
        owner: Team or service responsible for this asset.
        tags: Arbitrary key-value labels for discovery and filtering.
        classification: Data sensitivity classification level.
        schema_ref: Optional reference to a schema definition ID.
        description: Human-readable description of the asset's purpose.
        metadata: Additional unstructured metadata key-value pairs.
    """

    qualified_name: str = Field(
        ...,
        description="Fully qualified, globally unique asset name",
    )
    asset_type: AssetType = Field(
        ...,
        description="Classification of the asset's role in the data flow",
    )
    display_name: str = Field(
        default="",
        description="Human-readable display name for UIs and reports",
    )
    owner: str = Field(
        default="",
        description="Team or service responsible for this asset",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for discovery and filtering",
    )
    classification: AssetClassification = Field(
        default=AssetClassification.INTERNAL,
        description="Data sensitivity classification level",
    )
    schema_ref: Optional[str] = Field(
        None,
        description="Optional reference to a schema definition ID",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the asset's purpose",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional unstructured metadata key-value pairs",
    )

    model_config = {"extra": "forbid"}

    @field_validator("qualified_name")
    @classmethod
    def validate_qualified_name(cls, v: str) -> str:
        """Validate qualified_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("qualified_name must be non-empty")
        return v


class UpdateAssetRequest(BaseModel):
    """Request body for updating mutable fields of an existing data asset.

    Only fields explicitly included in this model can be updated.
    The qualified_name and asset_type are immutable once registered.
    All fields are optional; only provided fields will be updated.

    Attributes:
        display_name: Updated human-readable display name.
        owner: Updated team or service owner.
        tags: Updated key-value labels for discovery and filtering.
        classification: Updated data sensitivity classification.
        status: Updated lifecycle status (active, deprecated, archived).
        description: Updated human-readable description.
        metadata: Updated additional metadata key-value pairs.
    """

    display_name: Optional[str] = Field(
        None,
        description="Updated human-readable display name",
    )
    owner: Optional[str] = Field(
        None,
        description="Updated team or service owner",
    )
    tags: Optional[Dict[str, str]] = Field(
        None,
        description="Updated key-value labels for discovery and filtering",
    )
    classification: Optional[AssetClassification] = Field(
        None,
        description="Updated data sensitivity classification",
    )
    status: Optional[AssetStatus] = Field(
        None,
        description="Updated lifecycle status (active, deprecated, archived)",
    )
    description: Optional[str] = Field(
        None,
        description="Updated human-readable description",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated additional metadata key-value pairs",
    )

    model_config = {"extra": "forbid"}


class RecordTransformationRequest(BaseModel):
    """Request body for recording a data transformation event.

    Captures the details of a transformation performed by an agent or
    pipeline step, including source/target assets and record-level metrics.

    Attributes:
        transformation_type: Type of transformation performed.
        agent_id: ID of the agent that performed the transformation.
        pipeline_id: ID of the pipeline containing this transformation.
        execution_id: Unique identifier for this execution run.
        source_asset_ids: List of source asset IDs consumed.
        target_asset_ids: List of target asset IDs produced.
        description: Human-readable description of the transformation.
        parameters: Transformation-specific parameters.
        records_in: Number of input records consumed.
        records_out: Number of output records produced.
        records_filtered: Number of records removed by filtering.
        records_error: Number of records that failed transformation.
        duration_ms: Wall-clock duration in milliseconds.
    """

    transformation_type: TransformationType = Field(
        ...,
        description="Type of transformation performed",
    )
    agent_id: str = Field(
        default="",
        description="ID of the agent that performed the transformation",
    )
    pipeline_id: str = Field(
        default="",
        description="ID of the pipeline containing this transformation",
    )
    execution_id: str = Field(
        default="",
        description="Unique identifier for this execution run",
    )
    source_asset_ids: List[str] = Field(
        default_factory=list,
        description="List of source asset IDs consumed by the transform",
    )
    target_asset_ids: List[str] = Field(
        default_factory=list,
        description="List of target asset IDs produced by the transform",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the transformation",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transformation-specific parameters and configuration",
    )
    records_in: int = Field(
        default=0,
        ge=0,
        description="Number of input records consumed",
    )
    records_out: int = Field(
        default=0,
        ge=0,
        description="Number of output records produced",
    )
    records_filtered: int = Field(
        default=0,
        ge=0,
        description="Number of records removed by filtering",
    )
    records_error: int = Field(
        default=0,
        ge=0,
        description="Number of records that failed transformation",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration in milliseconds",
    )

    model_config = {"extra": "forbid"}


class CreateEdgeRequest(BaseModel):
    """Request body for creating a lineage edge between two assets.

    Attributes:
        source_asset_id: ID of the upstream source asset.
        target_asset_id: ID of the downstream target asset.
        transformation_id: Optional reference to a TransformationEvent ID.
        edge_type: Granularity level (dataset_level or column_level).
        source_field: Source field name (required for column_level edges).
        target_field: Target field name (required for column_level edges).
        transformation_logic: Description of the column-level transform logic.
        confidence: Confidence score for this edge (0.0 to 1.0).
    """

    source_asset_id: str = Field(
        ...,
        description="ID of the upstream source asset",
    )
    target_asset_id: str = Field(
        ...,
        description="ID of the downstream target asset",
    )
    transformation_id: Optional[str] = Field(
        None,
        description="Optional reference to a TransformationEvent ID",
    )
    edge_type: EdgeType = Field(
        default=EdgeType.DATASET_LEVEL,
        description="Granularity level (dataset_level or column_level)",
    )
    source_field: Optional[str] = Field(
        None,
        description="Source field name (required for column_level edges)",
    )
    target_field: Optional[str] = Field(
        None,
        description="Target field name (required for column_level edges)",
    )
    transformation_logic: Optional[str] = Field(
        None,
        description="Description of the column-level transformation logic",
    )
    confidence: float = Field(
        default=DEFAULT_EDGE_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Confidence score for this edge (0.0 to 1.0)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_asset_id")
    @classmethod
    def validate_source_asset_id(cls, v: str) -> str:
        """Validate source_asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_asset_id must be non-empty")
        return v

    @field_validator("target_asset_id")
    @classmethod
    def validate_target_asset_id(cls, v: str) -> str:
        """Validate target_asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_asset_id must be non-empty")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class RunImpactAnalysisRequest(BaseModel):
    """Request body for running impact analysis from a root asset.

    Triggers a graph traversal from the specified asset in the given
    direction to identify all affected assets within the max depth.

    Attributes:
        asset_id: ID of the root asset to analyze.
        direction: Direction of traversal (forward or backward).
        max_depth: Maximum traversal depth (default 10).
    """

    asset_id: str = Field(
        ...,
        description="ID of the root asset to analyze",
    )
    direction: TraversalDirection = Field(
        default=TraversalDirection.FORWARD,
        description="Direction of traversal (forward or backward)",
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        le=MAX_TRAVERSAL_DEPTH,
        description="Maximum traversal depth",
    )

    model_config = {"extra": "forbid"}

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("asset_id must be non-empty")
        return v


class RunValidationRequest(BaseModel):
    """Request body for running lineage graph validation.

    Triggers the validation engine to check the lineage graph for
    structural issues, coverage gaps, and freshness concerns.

    Attributes:
        scope: Scope of validation (e.g., "full", "namespace:xyz").
        include_freshness: Whether to include freshness scoring.
        include_coverage: Whether to include coverage analysis.
    """

    scope: str = Field(
        default="full",
        description="Scope of validation (e.g., 'full', 'namespace:xyz')",
    )
    include_freshness: bool = Field(
        default=True,
        description="Whether to include freshness scoring in the validation",
    )
    include_coverage: bool = Field(
        default=True,
        description="Whether to include coverage analysis in the validation",
    )

    model_config = {"extra": "forbid"}


class GenerateReportRequest(BaseModel):
    """Request body for generating a lineage report.

    Triggers the reporting engine to produce a lineage report in the
    specified format and type, scoped to the requested assets or
    the full graph.

    Attributes:
        report_type: Type of lineage report to generate.
        format: Output format for the report.
        scope: Scope of the report (e.g., "full", "asset:xyz").
        parameters: Report generation parameters and configuration.
        max_depth: Maximum traversal depth for report content.
    """

    report_type: ReportType = Field(
        default=ReportType.CUSTOM,
        description="Type of lineage report to generate",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format for the report",
    )
    scope: str = Field(
        default="full",
        description="Scope of the report (e.g., 'full', 'asset:xyz')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report generation parameters and configuration",
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        le=MAX_TRAVERSAL_DEPTH,
        description="Maximum traversal depth for report content",
    )

    model_config = {"extra": "forbid"}


class RunPipelineRequest(BaseModel):
    """Request body for running the full end-to-end data lineage pipeline.

    A single pipeline invocation encompasses all stages: asset registration,
    transformation capture, edge creation, validation, and optional report
    generation. Each stage can be individually enabled or disabled.

    Attributes:
        pipeline_id: Optional identifier for this pipeline run.
        scope: Scope of the pipeline (e.g., "full", "namespace:xyz").
        register_assets: Whether to register new assets during the run.
        capture_transformations: Whether to capture transformation events.
        run_validation: Whether to run validation after processing.
        generate_report: Whether to generate a lineage report.
        report_format: Format for the generated report (if enabled).
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional identifier for this pipeline run",
    )
    scope: str = Field(
        default="full",
        description="Scope of the pipeline (e.g., 'full', 'namespace:xyz')",
    )
    register_assets: bool = Field(
        default=True,
        description="Whether to register new assets during the run",
    )
    capture_transformations: bool = Field(
        default=True,
        description="Whether to capture transformation events",
    )
    run_validation: bool = Field(
        default=True,
        description="Whether to run validation after processing",
    )
    generate_report: bool = Field(
        default=False,
        description="Whether to generate a lineage report",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Format for the generated report (if enabled)",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (data_quality_profiler + cross_source_reconciliation)
    # -------------------------------------------------------------------------
    "QualityDimension",
    "SourceRegistryEngine",
    # -------------------------------------------------------------------------
    # Availability flags (for downstream feature detection)
    # -------------------------------------------------------------------------
    "_DQ_AVAILABLE",
    "_SR_AVAILABLE",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "VERSION",
    "MAX_ASSETS_PER_NAMESPACE",
    "MAX_TRAVERSAL_DEPTH",
    "MAX_EDGES_PER_ASSET",
    "DEFAULT_EDGE_CONFIDENCE",
    "MIN_IMPACT_CONFIDENCE",
    "MAX_CHANGE_EVENTS_PER_SNAPSHOT",
    "DEFAULT_PIPELINE_BATCH_SIZE",
    "SCORE_TIER_THRESHOLDS",
    "SUPPORTED_REPORT_FORMATS",
    "IMPACT_SEVERITY_ORDER",
    # -------------------------------------------------------------------------
    # Enumerations (14)
    # -------------------------------------------------------------------------
    "AssetType",
    "AssetClassification",
    "AssetStatus",
    "TransformationType",
    "EdgeType",
    "TraversalDirection",
    "ImpactSeverity",
    "ValidationResult",
    "ReportType",
    "ReportFormat",
    "ChangeType",
    "ChangeSeverity",
    "ScoreTier",
    "TransformationLogicType",
    # -------------------------------------------------------------------------
    # SDK data models (16)
    # -------------------------------------------------------------------------
    "DataAsset",
    "TransformationEvent",
    "LineageEdge",
    "GraphSnapshot",
    "ImpactAnalysisResult",
    "ValidationReport",
    "LineageReport",
    "ChangeEvent",
    "QualityScore",
    "AuditEntry",
    "GraphNode",
    "GraphEdgeView",
    "SubgraphResult",
    "LineageChain",
    "LineageStatistics",
    "PipelineResult",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "RegisterAssetRequest",
    "UpdateAssetRequest",
    "RecordTransformationRequest",
    "CreateEdgeRequest",
    "RunImpactAnalysisRequest",
    "RunValidationRequest",
    "GenerateReportRequest",
    "RunPipelineRequest",
]
