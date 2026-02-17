# -*- coding: utf-8 -*-
"""
Data Lineage Tracker Service Setup - AGENT-DATA-018

Provides ``configure_data_lineage_tracker(app)`` which wires up the
Data Lineage Tracker Agent SDK (asset registry, transformation tracker,
lineage graph, impact analyzer, lineage validator, lineage reporter,
pipeline orchestrator, provenance tracker) and mounts the REST API.

Also exposes ``get_data_lineage_tracker()`` for programmatic access
and the ``DataLineageTrackerService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.data_lineage_tracker.setup import configure_data_lineage_tracker
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_data_lineage_tracker(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.data_lineage_tracker.config import (
    DataLineageTrackerConfig,
    get_config,
)
from greenlang.data_lineage_tracker.metrics import (
    PROMETHEUS_AVAILABLE,
    record_asset_registered,
    record_transformation_captured,
    record_edge_created,
    record_impact_analysis,
    record_validation,
    record_report_generated,
    record_change_event,
    record_quality_score,
    observe_graph_traversal_duration,
    observe_processing_duration,
    set_graph_node_count,
    set_graph_edge_count,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.data_lineage_tracker.asset_registry import AssetRegistryEngine
except ImportError:
    AssetRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_lineage_tracker.transformation_tracker import TransformationTrackerEngine
except ImportError:
    TransformationTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
except ImportError:
    LineageGraphEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_lineage_tracker.impact_analyzer import ImpactAnalyzerEngine
except ImportError:
    ImpactAnalyzerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_lineage_tracker.lineage_validator import LineageValidatorEngine
except ImportError:
    LineageValidatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_lineage_tracker.lineage_reporter import LineageReporterEngine
except ImportError:
    LineageReporterEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_lineage_tracker.lineage_tracker_pipeline import LineageTrackerPipelineEngine
except ImportError:
    LineageTrackerPipelineEngine = None  # type: ignore[assignment, misc]


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class AssetResponse(BaseModel):
    """Data asset registration / retrieval response.

    Attributes:
        asset_id: Unique data asset identifier (UUID4).
        qualified_name: Fully qualified, globally unique asset name.
        asset_type: Classification of the asset role (dataset, field,
            agent, pipeline, report, metric, external_source).
        display_name: Human-readable display name.
        owner: Team or service responsible for this asset.
        tags: Key-value labels for discovery and filtering.
        classification: Data sensitivity level (public, internal,
            confidential, restricted).
        status: Lifecycle status (active, deprecated, archived).
        schema_ref: Optional reference to a schema definition ID.
        description: Human-readable description.
        metadata: Additional unstructured metadata.
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC last-update timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    qualified_name: str = Field(default="")
    asset_type: str = Field(default="dataset")
    display_name: str = Field(default="")
    owner: str = Field(default="")
    tags: Dict[str, str] = Field(default_factory=dict)
    classification: str = Field(default="internal")
    status: str = Field(default="active")
    schema_ref: Optional[str] = Field(default=None)
    description: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class TransformationResponse(BaseModel):
    """Transformation event capture response.

    Attributes:
        transformation_id: Unique transformation event identifier (UUID4).
        transformation_type: Type of transformation performed.
        agent_id: ID of the agent that performed the transformation.
        pipeline_id: ID of the pipeline containing this transformation.
        execution_id: Execution run identifier.
        source_asset_ids: List of source asset IDs consumed.
        target_asset_ids: List of target asset IDs produced.
        records_in: Number of input records consumed.
        records_out: Number of output records produced.
        records_filtered: Number of records removed by filtering.
        records_error: Number of records that failed transformation.
        duration_ms: Wall-clock duration in milliseconds.
        started_at: ISO-8601 UTC start timestamp.
        completed_at: ISO-8601 UTC completion timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    transformation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transformation_type: str = Field(default="filter")
    agent_id: str = Field(default="")
    pipeline_id: str = Field(default="")
    execution_id: str = Field(default="")
    source_asset_ids: List[str] = Field(default_factory=list)
    target_asset_ids: List[str] = Field(default_factory=list)
    records_in: int = Field(default=0)
    records_out: int = Field(default=0)
    records_filtered: int = Field(default=0)
    records_error: int = Field(default=0)
    duration_ms: float = Field(default=0.0)
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class EdgeResponse(BaseModel):
    """Lineage edge creation / retrieval response.

    Attributes:
        edge_id: Unique lineage edge identifier (UUID4).
        source_asset_id: ID of the upstream source asset.
        target_asset_id: ID of the downstream target asset.
        transformation_id: Optional reference to a TransformationEvent ID.
        edge_type: Granularity level (dataset_level or column_level).
        source_field: Source field name (for column-level edges).
        target_field: Target field name (for column-level edges).
        transformation_logic: Description of column-level transform.
        confidence: Confidence score (0.0 to 1.0).
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_asset_id: str = Field(default="")
    target_asset_id: str = Field(default="")
    transformation_id: Optional[str] = Field(default=None)
    edge_type: str = Field(default="dataset_level")
    source_field: Optional[str] = Field(default=None)
    target_field: Optional[str] = Field(default=None)
    transformation_logic: Optional[str] = Field(default=None)
    confidence: float = Field(default=1.0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class GraphResponse(BaseModel):
    """Full lineage graph snapshot response.

    Attributes:
        graph_id: Unique graph snapshot identifier (UUID4).
        node_count: Total number of asset nodes in the graph.
        edge_count: Total number of lineage edges in the graph.
        max_depth: Maximum depth of the longest lineage chain.
        connected_components: Number of disconnected subgraphs.
        orphan_count: Number of nodes with no edges.
        coverage_score: Lineage coverage fraction (0.0 to 1.0).
        graph_hash: SHA-256 hash of the graph topology.
        nodes: List of node detail dicts.
        edges: List of edge detail dicts.
        created_at: ISO-8601 UTC snapshot timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_count: int = Field(default=0)
    edge_count: int = Field(default=0)
    max_depth: int = Field(default=0)
    connected_components: int = Field(default=0)
    orphan_count: int = Field(default=0)
    coverage_score: float = Field(default=0.0)
    graph_hash: str = Field(default="")
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class SubgraphResponse(BaseModel):
    """Subgraph extraction result response.

    Attributes:
        root_asset_id: ID of the asset at the center of the subgraph.
        depth: Maximum traversal depth from the root.
        node_count: Total number of nodes in the subgraph.
        edge_count: Total number of edges in the subgraph.
        nodes: List of node detail dicts in the subgraph.
        edges: List of edge detail dicts in the subgraph.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    root_asset_id: str = Field(default="")
    depth: int = Field(default=0)
    node_count: int = Field(default=0)
    edge_count: int = Field(default=0)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ImpactAnalysisResponse(BaseModel):
    """Impact analysis traversal result response.

    Attributes:
        analysis_id: Unique analysis result identifier (UUID4).
        root_asset_id: ID of the asset from which traversal started.
        direction: Direction of graph traversal (forward or backward).
        depth: Maximum traversal depth reached.
        affected_assets: List of affected asset detail dicts.
        affected_assets_count: Total number of affected assets found.
        critical_count: Number of assets with CRITICAL severity.
        high_count: Number of assets with HIGH severity.
        medium_count: Number of assets with MEDIUM severity.
        low_count: Number of assets with LOW severity.
        blast_radius: Fraction of total graph nodes affected (0.0 to 1.0).
        created_at: ISO-8601 UTC analysis timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    root_asset_id: str = Field(default="")
    direction: str = Field(default="forward")
    depth: int = Field(default=0)
    affected_assets: List[Dict[str, Any]] = Field(default_factory=list)
    affected_assets_count: int = Field(default=0)
    critical_count: int = Field(default=0)
    high_count: int = Field(default=0)
    medium_count: int = Field(default=0)
    low_count: int = Field(default=0)
    blast_radius: float = Field(default=0.0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ValidationResponse(BaseModel):
    """Lineage graph validation result response.

    Attributes:
        validation_id: Unique validation report identifier (UUID4).
        scope: Scope of the validation (e.g. "full", "namespace:xyz").
        orphan_nodes: Number of nodes with no edges.
        broken_edges: Number of edges referencing non-existent nodes.
        cycles_detected: Number of cycles found in the directed graph.
        source_coverage: Fraction of sources with complete lineage.
        completeness_score: Overall lineage completeness score (0.0-1.0).
        freshness_score: Lineage freshness score (0.0-1.0).
        issues: List of specific validation issue dicts.
        recommendations: List of suggested remediation actions.
        result: Validation outcome (pass, warn, fail).
        validated_at: ISO-8601 UTC validation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scope: str = Field(default="full")
    orphan_nodes: int = Field(default=0)
    broken_edges: int = Field(default=0)
    cycles_detected: int = Field(default=0)
    source_coverage: float = Field(default=0.0)
    completeness_score: float = Field(default=0.0)
    freshness_score: float = Field(default=0.0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    result: str = Field(default="pass")
    validated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ReportResponse(BaseModel):
    """Lineage report generation result response.

    Attributes:
        report_id: Unique report identifier (UUID4).
        report_type: Type of lineage report (csrd_esrs, ghg_protocol,
            soc2, custom, visualization).
        format: Output format (mermaid, dot, json, d3, text, html, pdf).
        scope: Scope of the report.
        content: The rendered report content as a string.
        report_hash: SHA-256 hash of the report content.
        generated_by: Actor that requested the report.
        generated_at: ISO-8601 UTC generation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = Field(default="custom")
    format: str = Field(default="json")
    scope: str = Field(default="full")
    content: str = Field(default="")
    report_hash: str = Field(default="")
    generated_by: str = Field(default="system")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class PipelineResultResponse(BaseModel):
    """End-to-end lineage pipeline execution result response.

    Attributes:
        pipeline_id: Unique pipeline run identifier (UUID4).
        stages_completed: Number of pipeline stages that completed.
        assets_registered: Number of data assets registered.
        transformations_captured: Number of transformations recorded.
        edges_created: Number of lineage edges created.
        validation_result: Outcome of the validation stage.
        report_generated: Whether a lineage report was generated.
        duration_ms: Total wall-clock time in milliseconds.
        errors: List of error messages from failed stages.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stages_completed: int = Field(default=0)
    assets_registered: int = Field(default=0)
    transformations_captured: int = Field(default=0)
    edges_created: int = Field(default=0)
    validation_result: Optional[str] = Field(default=None)
    report_generated: bool = Field(default=False)
    duration_ms: float = Field(default=0.0)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DataLineageStatisticsResponse(BaseModel):
    """Aggregate statistics for the data lineage tracker service.

    Attributes:
        total_assets: Total number of registered data assets.
        total_transformations: Total number of recorded transformation events.
        total_edges: Total number of lineage edges in the graph.
        total_impact_analyses: Total number of impact analyses performed.
        total_validations: Total number of validation checks run.
        total_reports: Total number of lineage reports generated.
        avg_depth: Average lineage chain depth across all terminal assets.
        max_depth: Maximum lineage chain depth in the graph.
        coverage_score: Overall lineage coverage (0.0 to 1.0).
        orphan_count: Number of assets with no lineage connections.
        assets_by_type: Asset count broken down by AssetType.
    """

    model_config = {"extra": "forbid"}

    total_assets: int = Field(default=0)
    total_transformations: int = Field(default=0)
    total_edges: int = Field(default=0)
    total_impact_analyses: int = Field(default=0)
    total_validations: int = Field(default=0)
    total_reports: int = Field(default=0)
    avg_depth: float = Field(default=0.0)
    max_depth: int = Field(default=0)
    coverage_score: float = Field(default=0.0)
    orphan_count: int = Field(default=0)
    assets_by_type: Dict[str, int] = Field(default_factory=dict)


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# DataLineageTrackerService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["DataLineageTrackerService"] = None


class DataLineageTrackerService:
    """Unified facade over the Data Lineage Tracker Agent SDK.

    Aggregates all seven lineage engines (asset registry, transformation
    tracker, lineage graph, impact analyzer, lineage validator, lineage
    reporter, pipeline orchestrator) through a single entry point with
    convenience methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: DataLineageTrackerConfig instance.
        provenance: ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = DataLineageTrackerService()
        >>> result = service.register_asset(
        ...     qualified_name="emissions.scope3.spend_data",
        ...     asset_type="dataset",
        ...     display_name="Scope 3 Spend Data",
        ...     owner="data-team",
        ... )
        >>> print(result.asset_id, result.status)
    """

    def __init__(
        self,
        config: Optional[DataLineageTrackerConfig] = None,
    ) -> None:
        """Initialize the Data Lineage Tracker Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - AssetRegistryEngine
        - TransformationTrackerEngine
        - LineageGraphEngine
        - ImpactAnalyzerEngine
        - LineageValidatorEngine
        - LineageReporterEngine
        - LineageTrackerPipelineEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = ProvenanceTracker(
            genesis_hash=self.config.genesis_hash,
        )

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._asset_registry_engine: Any = None
        self._transformation_tracker_engine: Any = None
        self._lineage_graph_engine: Any = None
        self._impact_analyzer_engine: Any = None
        self._lineage_validator_engine: Any = None
        self._lineage_reporter_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._assets: Dict[str, AssetResponse] = {}
        self._transformations: Dict[str, TransformationResponse] = {}
        self._edges: Dict[str, EdgeResponse] = {}
        self._validations: Dict[str, ValidationResponse] = {}
        self._reports: Dict[str, ReportResponse] = {}
        self._impact_analyses: Dict[str, ImpactAnalysisResponse] = {}
        self._pipeline_results: Dict[str, PipelineResultResponse] = {}

        # Statistics
        self._stats = DataLineageStatisticsResponse()
        self._started = False

        logger.info("DataLineageTrackerService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def asset_registry_engine(self) -> Any:
        """Get the AssetRegistryEngine instance."""
        return self._asset_registry_engine

    @property
    def transformation_tracker_engine(self) -> Any:
        """Get the TransformationTrackerEngine instance."""
        return self._transformation_tracker_engine

    @property
    def lineage_graph_engine(self) -> Any:
        """Get the LineageGraphEngine instance."""
        return self._lineage_graph_engine

    @property
    def impact_analyzer_engine(self) -> Any:
        """Get the ImpactAnalyzerEngine instance."""
        return self._impact_analyzer_engine

    @property
    def lineage_validator_engine(self) -> Any:
        """Get the LineageValidatorEngine instance."""
        return self._lineage_validator_engine

    @property
    def lineage_reporter_engine(self) -> Any:
        """Get the LineageReporterEngine instance."""
        return self._lineage_reporter_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the LineageTrackerPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        if AssetRegistryEngine is not None:
            try:
                self._asset_registry_engine = AssetRegistryEngine(self.config)
                logger.info("AssetRegistryEngine initialized")
            except Exception as exc:
                logger.warning("AssetRegistryEngine init failed: %s", exc)
        else:
            logger.warning("AssetRegistryEngine not available; using stub")

        if TransformationTrackerEngine is not None:
            try:
                self._transformation_tracker_engine = TransformationTrackerEngine(
                    self.config,
                )
                logger.info("TransformationTrackerEngine initialized")
            except Exception as exc:
                logger.warning("TransformationTrackerEngine init failed: %s", exc)
        else:
            logger.warning("TransformationTrackerEngine not available; using stub")

        if LineageGraphEngine is not None:
            try:
                self._lineage_graph_engine = LineageGraphEngine(self.config)
                logger.info("LineageGraphEngine initialized")
            except Exception as exc:
                logger.warning("LineageGraphEngine init failed: %s", exc)
        else:
            logger.warning("LineageGraphEngine not available; using stub")

        if ImpactAnalyzerEngine is not None:
            try:
                self._impact_analyzer_engine = ImpactAnalyzerEngine(self.config)
                logger.info("ImpactAnalyzerEngine initialized")
            except Exception as exc:
                logger.warning("ImpactAnalyzerEngine init failed: %s", exc)
        else:
            logger.warning("ImpactAnalyzerEngine not available; using stub")

        if LineageValidatorEngine is not None:
            try:
                self._lineage_validator_engine = LineageValidatorEngine(self.config)
                logger.info("LineageValidatorEngine initialized")
            except Exception as exc:
                logger.warning("LineageValidatorEngine init failed: %s", exc)
        else:
            logger.warning("LineageValidatorEngine not available; using stub")

        if LineageReporterEngine is not None:
            try:
                self._lineage_reporter_engine = LineageReporterEngine(self.config)
                logger.info("LineageReporterEngine initialized")
            except Exception as exc:
                logger.warning("LineageReporterEngine init failed: %s", exc)
        else:
            logger.warning("LineageReporterEngine not available; using stub")

        if LineageTrackerPipelineEngine is not None:
            try:
                self._pipeline_engine = LineageTrackerPipelineEngine(self.config)
                logger.info("LineageTrackerPipelineEngine initialized")
            except Exception as exc:
                logger.warning("LineageTrackerPipelineEngine init failed: %s", exc)
        else:
            logger.warning("LineageTrackerPipelineEngine not available; using stub")

    # ==================================================================
    # Asset operations (delegate to AssetRegistryEngine)
    # ==================================================================

    def register_asset(
        self,
        qualified_name: str,
        asset_type: str,
        display_name: str = "",
        owner: str = "",
        tags: Optional[Dict[str, str]] = None,
        classification: str = "internal",
        schema_ref: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssetResponse:
        """Register a new data asset in the lineage graph.

        Delegates to the AssetRegistryEngine for registration. All
        operations are deterministic. No LLM is used for registration
        logic (zero-hallucination).

        Args:
            qualified_name: Fully qualified, globally unique asset name
                (e.g. ``"erp.sap.spend_data.amount_usd"``).
            asset_type: Classification of asset role (``"dataset"``,
                ``"field"``, ``"agent"``, ``"pipeline"``, ``"report"``,
                ``"metric"``, ``"external_source"``).
            display_name: Human-readable display name.
            owner: Team or service responsible for this asset.
            tags: Key-value labels for discovery and filtering.
            classification: Data sensitivity level.
            schema_ref: Optional reference to a schema definition ID.
            description: Human-readable description.
            metadata: Additional unstructured metadata.

        Returns:
            AssetResponse with registered asset details.

        Raises:
            ValueError: If qualified_name or asset_type are empty.
        """
        t0 = time.perf_counter()

        if not qualified_name:
            raise ValueError("qualified_name must not be empty")
        if not asset_type:
            raise ValueError("asset_type must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._asset_registry_engine is not None:
                engine_result = self._asset_registry_engine.register_asset(
                    qualified_name=qualified_name,
                    asset_type=asset_type,
                    display_name=display_name,
                    owner=owner,
                    tags=tags,
                    classification=classification,
                    schema_ref=schema_ref,
                    description=description,
                    metadata=metadata,
                )

            # Build response
            asset_id = (
                engine_result.get("asset_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = AssetResponse(
                asset_id=asset_id,
                qualified_name=qualified_name,
                asset_type=asset_type,
                display_name=display_name or qualified_name,
                owner=owner,
                tags=tags or {},
                classification=classification,
                status=engine_result.get("status", "active") if engine_result else "active",
                schema_ref=schema_ref,
                description=description,
                metadata=metadata or {},
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
                updated_at=engine_result.get("updated_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._assets[response.asset_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="lineage_asset",
                entity_id=response.asset_id,
                action="asset_registered",
                data={
                    "qualified_name": qualified_name,
                    "asset_type": asset_type,
                    "owner": owner,
                    "classification": classification,
                },
            )

            # Record metrics
            record_asset_registered(asset_type, classification)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("asset_register", elapsed)

            # Update statistics
            self._stats.total_assets += 1
            type_counts = self._stats.assets_by_type
            type_counts[asset_type] = type_counts.get(asset_type, 0) + 1

            # Update gauge
            set_graph_node_count(self._stats.total_assets)

            logger.info(
                "Registered asset %s: qualified_name=%s, type=%s",
                response.asset_id,
                qualified_name,
                asset_type,
            )
            return response

        except Exception as exc:
            logger.error("register_asset failed: %s", exc, exc_info=True)
            raise

    def get_asset(self, asset_id: str) -> Optional[AssetResponse]:
        """Get a data asset by its unique identifier.

        Args:
            asset_id: Asset identifier (UUID4 string).

        Returns:
            AssetResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._asset_registry_engine is not None:
                engine_result = self._asset_registry_engine.get_asset(asset_id)
                if engine_result is not None:
                    resp = self._dict_to_asset_response(engine_result)
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("asset_get", elapsed)
                    return resp
                return None

            # Fallback to in-memory store
            result = self._assets.get(asset_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("asset_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_asset failed: %s", exc, exc_info=True)
            raise

    def update_asset(
        self,
        asset_id: str,
        display_name: Optional[str] = None,
        owner: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        classification: Optional[str] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AssetResponse]:
        """Update mutable fields of an existing data asset.

        Only fields explicitly provided (non-None) are updated. The
        qualified_name and asset_type are immutable after registration.

        Args:
            asset_id: Asset identifier to update.
            display_name: Updated display name, or None to leave unchanged.
            owner: Updated owner, or None to leave unchanged.
            tags: Updated tags, or None to leave unchanged.
            classification: Updated classification, or None to leave unchanged.
            status: Updated lifecycle status, or None to leave unchanged.
            description: Updated description, or None to leave unchanged.
            metadata: Updated metadata, or None to leave unchanged.

        Returns:
            Updated AssetResponse or None if asset not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._asset_registry_engine is not None:
                try:
                    engine_result = self._asset_registry_engine.update_asset(
                        asset_id=asset_id,
                        display_name=display_name,
                        owner=owner,
                        tags=tags,
                        classification=classification,
                        status=status,
                        description=description,
                        metadata=metadata,
                    )
                except KeyError:
                    return None

            if engine_result is not None:
                response = self._dict_to_asset_response(engine_result)
                response.provenance_hash = _compute_hash(response)
                self._assets[response.asset_id] = response
            else:
                # Fallback to in-memory store
                cached = self._assets.get(asset_id)
                if cached is None:
                    return None
                if display_name is not None:
                    cached.display_name = display_name
                if owner is not None:
                    cached.owner = owner
                if tags is not None:
                    cached.tags = tags
                if classification is not None:
                    cached.classification = classification
                if status is not None:
                    cached.status = status
                if description is not None:
                    cached.description = description
                if metadata is not None:
                    cached.metadata = metadata
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
                response = cached

            # Record provenance
            self.provenance.record(
                entity_type="lineage_asset",
                entity_id=asset_id,
                action="asset_updated",
                data={
                    "display_name": display_name,
                    "owner": owner,
                    "status": status,
                    "classification": classification,
                },
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("asset_update", elapsed)

            logger.info(
                "Updated asset %s: owner=%s, status=%s",
                asset_id,
                owner,
                status,
            )
            return response

        except Exception as exc:
            logger.error("update_asset failed: %s", exc, exc_info=True)
            raise

    def delete_asset(self, asset_id: str) -> bool:
        """Soft-delete a data asset by setting its status to archived.

        Args:
            asset_id: Asset identifier to delete.

        Returns:
            True if the asset was found and archived, False if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._asset_registry_engine is not None:
                try:
                    self._asset_registry_engine.delete_asset(asset_id)
                except (KeyError, ValueError):
                    return False

            # Update in-memory cache
            cached = self._assets.get(asset_id)
            if cached is not None:
                cached.status = "archived"
                cached.updated_at = _utcnow_iso()
                cached.provenance_hash = _compute_hash(cached)
            elif self._asset_registry_engine is None:
                return False

            # Record provenance
            self.provenance.record(
                entity_type="lineage_asset",
                entity_id=asset_id,
                action="asset_deleted",
            )

            elapsed = time.perf_counter() - t0
            observe_processing_duration("asset_delete", elapsed)

            logger.info("Soft-deleted (archived) asset %s", asset_id)
            return True

        except Exception as exc:
            logger.error("delete_asset failed: %s", exc, exc_info=True)
            raise

    def search_assets(
        self,
        asset_type: Optional[str] = None,
        owner: Optional[str] = None,
        classification: Optional[str] = None,
        status: Optional[str] = None,
        tag_key: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AssetResponse]:
        """Search assets with optional filtering and pagination.

        All filters are applied with AND logic.

        Args:
            asset_type: Filter by exact asset type.
            owner: Filter by exact owner.
            classification: Filter by exact classification.
            status: Filter by exact lifecycle status.
            tag_key: Filter by tag key presence.
            query: Substring search in qualified_name and display_name.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of AssetResponse instances matching the filters.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._asset_registry_engine is not None:
                try:
                    engine_results = self._asset_registry_engine.search_assets(
                        asset_type=asset_type,
                        owner=owner,
                        classification=classification,
                        status=status,
                        tag_key=tag_key,
                        query=query,
                        limit=limit,
                        offset=offset,
                    )
                    results = [
                        self._dict_to_asset_response(rec)
                        for rec in engine_results
                    ]
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("asset_search", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            assets = list(self._assets.values())
            filtered = self._filter_assets(
                assets, asset_type, owner, classification, status,
                tag_key, query,
            )
            paginated = filtered[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("asset_search", elapsed)
            return paginated

        except Exception as exc:
            logger.error("search_assets failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Transformation operations (delegate to TransformationTrackerEngine)
    # ==================================================================

    def record_transformation(
        self,
        transformation_type: str,
        agent_id: str = "",
        pipeline_id: str = "",
        execution_id: str = "",
        source_asset_ids: Optional[List[str]] = None,
        target_asset_ids: Optional[List[str]] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        records_in: int = 0,
        records_out: int = 0,
        records_filtered: int = 0,
        records_error: int = 0,
        duration_ms: float = 0.0,
    ) -> TransformationResponse:
        """Record a data transformation event in the lineage graph.

        Captures the details of a transformation performed by an agent
        or pipeline step. All operations are deterministic (zero-hallucination).

        Args:
            transformation_type: Type of transformation performed.
            agent_id: ID of the agent that performed the transformation.
            pipeline_id: ID of the pipeline containing this transformation.
            execution_id: Unique identifier for this execution run.
            source_asset_ids: List of source asset IDs consumed.
            target_asset_ids: List of target asset IDs produced.
            description: Human-readable description.
            parameters: Transformation-specific parameters.
            records_in: Number of input records consumed.
            records_out: Number of output records produced.
            records_filtered: Number of records removed by filtering.
            records_error: Number of records that failed transformation.
            duration_ms: Wall-clock duration in milliseconds.

        Returns:
            TransformationResponse with recorded transformation details.

        Raises:
            ValueError: If transformation_type is empty.
        """
        t0 = time.perf_counter()

        if not transformation_type:
            raise ValueError("transformation_type must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._transformation_tracker_engine is not None:
                engine_result = self._transformation_tracker_engine.record_transformation(
                    transformation_type=transformation_type,
                    agent_id=agent_id,
                    pipeline_id=pipeline_id,
                    execution_id=execution_id or _new_uuid(),
                    source_asset_ids=source_asset_ids or [],
                    target_asset_ids=target_asset_ids or [],
                    description=description,
                    parameters=parameters or {},
                    records_in=records_in,
                    records_out=records_out,
                    records_filtered=records_filtered,
                    records_error=records_error,
                    duration_ms=duration_ms,
                )

            # Build response
            txn_id = (
                engine_result.get("id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = TransformationResponse(
                transformation_id=txn_id,
                transformation_type=transformation_type,
                agent_id=agent_id,
                pipeline_id=pipeline_id,
                execution_id=execution_id or txn_id,
                source_asset_ids=source_asset_ids or [],
                target_asset_ids=target_asset_ids or [],
                records_in=records_in,
                records_out=records_out,
                records_filtered=records_filtered,
                records_error=records_error,
                duration_ms=duration_ms,
                started_at=engine_result.get("started_at", now_iso) if engine_result else now_iso,
                completed_at=engine_result.get("completed_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._transformations[response.transformation_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="transformation",
                entity_id=response.transformation_id,
                action="transformation_captured",
                data={
                    "transformation_type": transformation_type,
                    "agent_id": agent_id,
                    "records_in": records_in,
                    "records_out": records_out,
                },
            )

            # Record metrics
            record_transformation_captured(transformation_type, agent_id or "unknown")
            elapsed = time.perf_counter() - t0
            observe_processing_duration("transformation_capture", elapsed)

            # Update statistics
            self._stats.total_transformations += 1

            logger.info(
                "Recorded transformation %s: type=%s agent=%s in=%d out=%d",
                response.transformation_id,
                transformation_type,
                agent_id,
                records_in,
                records_out,
            )
            return response

        except Exception as exc:
            logger.error("record_transformation failed: %s", exc, exc_info=True)
            raise

    def list_transformations(
        self,
        agent_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        transformation_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TransformationResponse]:
        """List transformation events with optional filtering.

        Args:
            agent_id: Filter by exact agent identifier.
            pipeline_id: Filter by exact pipeline identifier.
            transformation_type: Filter by exact transformation type.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of TransformationResponse instances.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._transformation_tracker_engine is not None:
                try:
                    engine_results = self._transformation_tracker_engine.list_transformations(
                        agent_id=agent_id,
                        pipeline_id=pipeline_id,
                        transformation_type=transformation_type,
                        limit=limit,
                        offset=offset,
                    )
                    results = [
                        self._dict_to_transformation_response(rec)
                        for rec in engine_results
                    ]
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("transformation_list", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            txns = list(self._transformations.values())
            if agent_id is not None:
                txns = [t for t in txns if t.agent_id == agent_id]
            if pipeline_id is not None:
                txns = [t for t in txns if t.pipeline_id == pipeline_id]
            if transformation_type is not None:
                txns = [t for t in txns if t.transformation_type == transformation_type]
            paginated = txns[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("transformation_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_transformations failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Edge operations (delegate to LineageGraphEngine)
    # ==================================================================

    def create_edge(
        self,
        source_asset_id: str,
        target_asset_id: str,
        transformation_id: Optional[str] = None,
        edge_type: str = "dataset_level",
        source_field: Optional[str] = None,
        target_field: Optional[str] = None,
        transformation_logic: Optional[str] = None,
        confidence: float = 1.0,
    ) -> EdgeResponse:
        """Create a lineage edge between two data assets.

        Delegates to the LineageGraphEngine for edge creation. All
        operations are deterministic (zero-hallucination).

        Args:
            source_asset_id: ID of the upstream source asset.
            target_asset_id: ID of the downstream target asset.
            transformation_id: Optional reference to a TransformationEvent.
            edge_type: Granularity level (dataset_level or column_level).
            source_field: Source field name (for column-level edges).
            target_field: Target field name (for column-level edges).
            transformation_logic: Column-level transform description.
            confidence: Confidence score for this edge (0.0 to 1.0).

        Returns:
            EdgeResponse with created edge details.

        Raises:
            ValueError: If source_asset_id or target_asset_id are empty.
        """
        t0 = time.perf_counter()

        if not source_asset_id:
            raise ValueError("source_asset_id must not be empty")
        if not target_asset_id:
            raise ValueError("target_asset_id must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._lineage_graph_engine is not None:
                engine_result = self._lineage_graph_engine.add_edge(
                    source_asset_id,
                    target_asset_id,
                    edge_type=edge_type,
                    transformation_id=transformation_id,
                    source_field=source_field,
                    target_field=target_field,
                    transformation_logic=transformation_logic,
                    confidence=confidence,
                )

            # Build response
            edge_id = (
                engine_result.get("edge_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = EdgeResponse(
                edge_id=edge_id,
                source_asset_id=source_asset_id,
                target_asset_id=target_asset_id,
                transformation_id=transformation_id,
                edge_type=edge_type,
                source_field=source_field,
                target_field=target_field,
                transformation_logic=transformation_logic,
                confidence=confidence,
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._edges[response.edge_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="lineage_edge",
                entity_id=response.edge_id,
                action="edge_created",
                data={
                    "source_asset_id": source_asset_id,
                    "target_asset_id": target_asset_id,
                    "edge_type": edge_type,
                    "confidence": confidence,
                },
            )

            # Record metrics
            record_edge_created(edge_type)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("edge_create", elapsed)

            # Update statistics
            self._stats.total_edges += 1
            set_graph_edge_count(self._stats.total_edges)

            logger.info(
                "Created edge %s: %s -> %s type=%s confidence=%.2f",
                response.edge_id,
                source_asset_id,
                target_asset_id,
                edge_type,
                confidence,
            )
            return response

        except Exception as exc:
            logger.error("create_edge failed: %s", exc, exc_info=True)
            raise

    def list_edges(
        self,
        source_asset_id: Optional[str] = None,
        target_asset_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[EdgeResponse]:
        """List lineage edges with optional filtering.

        Args:
            source_asset_id: Filter by exact source asset ID.
            target_asset_id: Filter by exact target asset ID.
            edge_type: Filter by exact edge type.
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of EdgeResponse instances.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._lineage_graph_engine is not None:
                try:
                    engine_results = self._lineage_graph_engine.list_edges(
                        source_asset_id=source_asset_id,
                        target_asset_id=target_asset_id,
                        edge_type=edge_type,
                        limit=limit,
                        offset=offset,
                    )
                    results = [
                        self._dict_to_edge_response(rec)
                        for rec in engine_results
                    ]
                    elapsed = time.perf_counter() - t0
                    observe_processing_duration("edge_list", elapsed)
                    return results
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store with filtering
            edges = list(self._edges.values())
            if source_asset_id is not None:
                edges = [e for e in edges if e.source_asset_id == source_asset_id]
            if target_asset_id is not None:
                edges = [e for e in edges if e.target_asset_id == target_asset_id]
            if edge_type is not None:
                edges = [e for e in edges if e.edge_type == edge_type]
            paginated = edges[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            observe_processing_duration("edge_list", elapsed)
            return paginated

        except Exception as exc:
            logger.error("list_edges failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Graph operations (delegate to LineageGraphEngine)
    # ==================================================================

    def get_graph(self) -> GraphResponse:
        """Get the full lineage graph snapshot.

        Returns:
            GraphResponse with full graph topology and statistics.
        """
        t0 = time.perf_counter()

        try:
            engine_result: Optional[Dict[str, Any]] = None
            if self._lineage_graph_engine is not None:
                try:
                    engine_result = self._lineage_graph_engine.get_snapshot()
                except (AttributeError, TypeError):
                    pass

            node_count = engine_result.get("node_count", 0) if engine_result else len(self._assets)
            edge_count = engine_result.get("edge_count", 0) if engine_result else len(self._edges)
            max_depth = engine_result.get("max_depth", 0) if engine_result else 0
            components = engine_result.get("connected_components", 0) if engine_result else 0
            orphans = engine_result.get("orphan_count", 0) if engine_result else 0
            coverage = engine_result.get("coverage_score", 0.0) if engine_result else 0.0
            graph_hash = engine_result.get("graph_hash", "") if engine_result else ""
            nodes = engine_result.get("nodes", []) if engine_result else []
            edges = engine_result.get("edges", []) if engine_result else []

            response = GraphResponse(
                node_count=node_count,
                edge_count=edge_count,
                max_depth=max_depth,
                connected_components=components,
                orphan_count=orphans,
                coverage_score=coverage,
                graph_hash=graph_hash,
                nodes=nodes,
                edges=edges,
                created_at=_utcnow_iso(),
            )
            response.provenance_hash = _compute_hash(response)

            elapsed = time.perf_counter() - t0
            observe_processing_duration("graph_get", elapsed)

            logger.info(
                "Graph snapshot: nodes=%d edges=%d depth=%d coverage=%.2f",
                node_count,
                edge_count,
                max_depth,
                coverage,
            )
            return response

        except Exception as exc:
            logger.error("get_graph failed: %s", exc, exc_info=True)
            raise

    def get_subgraph(
        self,
        root_asset_id: str,
        depth: int = 10,
        direction: str = "both",
    ) -> SubgraphResponse:
        """Extract a subgraph around a root asset.

        Args:
            root_asset_id: ID of the asset at the center of the subgraph.
            depth: Maximum traversal depth from the root.
            direction: Traversal direction (forward, backward, both).

        Returns:
            SubgraphResponse with the extracted subgraph.

        Raises:
            ValueError: If root_asset_id is empty.
        """
        t0 = time.perf_counter()

        if not root_asset_id:
            raise ValueError("root_asset_id must not be empty")

        try:
            engine_result: Optional[Dict[str, Any]] = None
            if self._lineage_graph_engine is not None:
                try:
                    engine_result = self._lineage_graph_engine.get_subgraph(
                        root_asset_id,
                        max_depth=depth,
                        direction=direction,
                    )
                except (AttributeError, TypeError):
                    pass

            nodes = engine_result.get("nodes", []) if engine_result else []
            edges = engine_result.get("edges", []) if engine_result else []

            response = SubgraphResponse(
                root_asset_id=root_asset_id,
                depth=depth,
                node_count=len(nodes),
                edge_count=len(edges),
                nodes=nodes,
                edges=edges,
            )
            response.provenance_hash = _compute_hash(response)

            elapsed = time.perf_counter() - t0
            observe_graph_traversal_duration(elapsed)
            observe_processing_duration("graph_subgraph", elapsed)

            logger.info(
                "Subgraph for %s: depth=%d nodes=%d edges=%d",
                root_asset_id,
                depth,
                len(nodes),
                len(edges),
            )
            return response

        except Exception as exc:
            logger.error("get_subgraph failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Impact analysis (delegate to ImpactAnalyzerEngine)
    # ==================================================================

    def analyze_backward(
        self,
        asset_id: str,
        max_depth: int = 10,
    ) -> ImpactAnalysisResponse:
        """Perform backward lineage traversal (provenance) from an asset.

        Traces the asset upstream to all its source datasets, fields,
        and external sources. All traversal logic is deterministic BFS
        (zero-hallucination).

        Args:
            asset_id: ID of the asset to trace backwards from.
            max_depth: Maximum traversal depth.

        Returns:
            ImpactAnalysisResponse with upstream provenance results.

        Raises:
            ValueError: If asset_id is empty.
        """
        t0 = time.perf_counter()

        if not asset_id:
            raise ValueError("asset_id must not be empty")

        try:
            engine_result: Optional[Dict[str, Any]] = None
            if self._impact_analyzer_engine is not None:
                engine_result = self._impact_analyzer_engine.analyze_backward(
                    asset_id,
                    max_depth=max_depth,
                )

            response = self._build_impact_response(
                engine_result, asset_id, "backward", max_depth,
            )

            # Record provenance
            self.provenance.record(
                entity_type="impact_analysis",
                entity_id=response.analysis_id,
                action="backward_analysis",
                data={
                    "asset_id": asset_id,
                    "direction": "backward",
                    "affected_count": response.affected_assets_count,
                },
            )

            # Record metrics
            severity = self._highest_severity(response)
            record_impact_analysis("backward", severity)
            elapsed = time.perf_counter() - t0
            observe_graph_traversal_duration(elapsed)
            observe_processing_duration("impact_backward", elapsed)

            # Update statistics
            self._stats.total_impact_analyses += 1

            logger.info(
                "Backward analysis from %s: affected=%d depth=%d",
                asset_id,
                response.affected_assets_count,
                response.depth,
            )
            return response

        except Exception as exc:
            logger.error("analyze_backward failed: %s", exc, exc_info=True)
            raise

    def analyze_forward(
        self,
        asset_id: str,
        max_depth: int = 10,
    ) -> ImpactAnalysisResponse:
        """Perform forward impact analysis from an asset.

        Traces the asset downstream to all its consumers to assess the
        blast radius. All traversal logic is deterministic BFS
        (zero-hallucination).

        Args:
            asset_id: ID of the asset to trace forward from.
            max_depth: Maximum traversal depth.

        Returns:
            ImpactAnalysisResponse with downstream impact results.

        Raises:
            ValueError: If asset_id is empty.
        """
        t0 = time.perf_counter()

        if not asset_id:
            raise ValueError("asset_id must not be empty")

        try:
            engine_result: Optional[Dict[str, Any]] = None
            if self._impact_analyzer_engine is not None:
                engine_result = self._impact_analyzer_engine.analyze_forward(
                    asset_id,
                    max_depth=max_depth,
                )

            response = self._build_impact_response(
                engine_result, asset_id, "forward", max_depth,
            )

            # Record provenance
            self.provenance.record(
                entity_type="impact_analysis",
                entity_id=response.analysis_id,
                action="forward_analysis",
                data={
                    "asset_id": asset_id,
                    "direction": "forward",
                    "affected_count": response.affected_assets_count,
                    "blast_radius": response.blast_radius,
                },
            )

            # Record metrics
            severity = self._highest_severity(response)
            record_impact_analysis("forward", severity)
            elapsed = time.perf_counter() - t0
            observe_graph_traversal_duration(elapsed)
            observe_processing_duration("impact_forward", elapsed)

            # Update statistics
            self._stats.total_impact_analyses += 1

            logger.info(
                "Forward analysis from %s: affected=%d blast_radius=%.2f",
                asset_id,
                response.affected_assets_count,
                response.blast_radius,
            )
            return response

        except Exception as exc:
            logger.error("analyze_forward failed: %s", exc, exc_info=True)
            raise

    def run_impact_analysis(
        self,
        asset_id: str,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> ImpactAnalysisResponse:
        """Run impact analysis in the specified direction.

        Convenience method that dispatches to analyze_forward or
        analyze_backward based on the direction parameter.

        Args:
            asset_id: ID of the root asset to analyze.
            direction: Direction of traversal (forward or backward).
            max_depth: Maximum traversal depth.

        Returns:
            ImpactAnalysisResponse with analysis results.

        Raises:
            ValueError: If asset_id is empty or direction is invalid.
        """
        if direction == "backward":
            return self.analyze_backward(asset_id, max_depth=max_depth)
        return self.analyze_forward(asset_id, max_depth=max_depth)

    # ==================================================================
    # Validation (delegate to LineageValidatorEngine)
    # ==================================================================

    def validate_lineage(
        self,
        scope: str = "full",
        include_freshness: bool = True,
        include_coverage: bool = True,
    ) -> ValidationResponse:
        """Run lineage graph validation and health check.

        Checks the lineage graph for structural issues such as orphan
        nodes, broken edges, cycles, and coverage gaps. All validation
        logic is deterministic (zero-hallucination).

        Args:
            scope: Scope of the validation (e.g. "full", "namespace:xyz").
            include_freshness: Whether to include freshness scoring.
            include_coverage: Whether to include coverage analysis.

        Returns:
            ValidationResponse with validation results.
        """
        t0 = time.perf_counter()

        try:
            engine_result: Optional[Dict[str, Any]] = None
            if self._lineage_validator_engine is not None:
                engine_result = self._lineage_validator_engine.validate(
                    scope=scope,
                    include_freshness=include_freshness,
                    include_coverage=include_coverage,
                )

            # Build response
            validation_id = (
                engine_result.get("id", _new_uuid())
                if engine_result else _new_uuid()
            )
            orphan_nodes = engine_result.get("orphan_nodes", 0) if engine_result else 0
            broken_edges = engine_result.get("broken_edges", 0) if engine_result else 0
            cycles = engine_result.get("cycles_detected", 0) if engine_result else 0
            src_coverage = engine_result.get("source_coverage", 0.0) if engine_result else 0.0
            completeness = engine_result.get("completeness_score", 0.0) if engine_result else 0.0
            freshness = engine_result.get("freshness_score", 0.0) if engine_result else 0.0
            issues = engine_result.get("issues", []) if engine_result else []
            recs = engine_result.get("recommendations", []) if engine_result else []

            # Determine result
            if broken_edges > 0 or cycles > 0:
                result_str = "fail"
            elif orphan_nodes > 0 or src_coverage < 0.8:
                result_str = "warn"
            else:
                result_str = "pass"

            if engine_result and "result" in engine_result:
                result_str = engine_result["result"]

            response = ValidationResponse(
                validation_id=validation_id,
                scope=scope,
                orphan_nodes=orphan_nodes,
                broken_edges=broken_edges,
                cycles_detected=cycles,
                source_coverage=src_coverage,
                completeness_score=completeness,
                freshness_score=freshness,
                issues=issues,
                recommendations=recs,
                result=result_str,
                validated_at=_utcnow_iso(),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._validations[response.validation_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="validation",
                entity_id=response.validation_id,
                action="validation_completed",
                data={
                    "scope": scope,
                    "result": result_str,
                    "orphan_nodes": orphan_nodes,
                    "broken_edges": broken_edges,
                    "cycles_detected": cycles,
                },
            )

            # Record metrics
            record_validation(result_str)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("validate", elapsed)

            # Update statistics
            self._stats.total_validations += 1

            logger.info(
                "Validation %s: scope=%s result=%s orphans=%d broken=%d cycles=%d",
                response.validation_id,
                scope,
                result_str,
                orphan_nodes,
                broken_edges,
                cycles,
            )
            return response

        except Exception as exc:
            logger.error("validate_lineage failed: %s", exc, exc_info=True)
            raise

    def get_validation(
        self,
        validation_id: str,
    ) -> Optional[ValidationResponse]:
        """Get a validation result by its identifier.

        Args:
            validation_id: Validation report identifier.

        Returns:
            ValidationResponse or None if not found.
        """
        t0 = time.perf_counter()

        try:
            result = self._validations.get(validation_id)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("validation_get", elapsed)
            return result

        except Exception as exc:
            logger.error("get_validation failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Report generation (delegate to LineageReporterEngine)
    # ==================================================================

    def generate_report(
        self,
        report_type: str = "custom",
        report_format: str = "json",
        scope: str = "full",
        parameters: Optional[Dict[str, Any]] = None,
        max_depth: int = 10,
    ) -> ReportResponse:
        """Generate a lineage report in the specified format.

        Delegates to the LineageReporterEngine. All report generation
        logic is deterministic (zero-hallucination).

        Args:
            report_type: Type of lineage report (csrd_esrs, ghg_protocol,
                soc2, custom, visualization).
            report_format: Output format (mermaid, dot, json, d3, text,
                html, pdf).
            scope: Scope of the report.
            parameters: Report generation parameters.
            max_depth: Maximum traversal depth for report content.

        Returns:
            ReportResponse with generated report content.
        """
        t0 = time.perf_counter()

        try:
            engine_result: Optional[Dict[str, Any]] = None
            if self._lineage_reporter_engine is not None:
                engine_result = self._lineage_reporter_engine.generate_report(
                    report_type=report_type,
                    report_format=report_format,
                    scope=scope,
                    parameters=parameters or {},
                    max_depth=max_depth,
                )

            # Build response
            report_id = (
                engine_result.get("id", _new_uuid())
                if engine_result else _new_uuid()
            )
            content = engine_result.get("content", "") if engine_result else ""
            report_hash = (
                engine_result.get("report_hash", "")
                if engine_result else ""
            )
            if not report_hash and content:
                report_hash = hashlib.sha256(content.encode()).hexdigest()

            response = ReportResponse(
                report_id=report_id,
                report_type=report_type,
                format=report_format,
                scope=scope,
                content=content,
                report_hash=report_hash,
                generated_at=_utcnow_iso(),
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._reports[response.report_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="report",
                entity_id=response.report_id,
                action="report_generated",
                data={
                    "report_type": report_type,
                    "format": report_format,
                    "scope": scope,
                },
            )

            # Record metrics
            record_report_generated(report_type, report_format)
            elapsed = time.perf_counter() - t0
            observe_processing_duration("report_generate", elapsed)

            # Update statistics
            self._stats.total_reports += 1

            logger.info(
                "Generated report %s: type=%s format=%s scope=%s",
                response.report_id,
                report_type,
                report_format,
                scope,
            )
            return response

        except Exception as exc:
            logger.error("generate_report failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Pipeline orchestration (delegate to LineageTrackerPipelineEngine)
    # ==================================================================

    def run_pipeline(
        self,
        scope: str = "full",
        register_assets: bool = True,
        capture_transformations: bool = True,
        validate: bool = True,
        generate_report: bool = False,
        report_format: str = "json",
    ) -> PipelineResultResponse:
        """Run the end-to-end data lineage pipeline.

        Orchestrates all stages: asset registration, transformation
        capture, edge creation, validation, and optional report
        generation.

        Args:
            scope: Scope of the pipeline.
            register_assets: Whether to register new assets.
            capture_transformations: Whether to capture transformations.
            validate: Whether to run validation after processing.
            generate_report: Whether to generate a lineage report.
            report_format: Format for the generated report.

        Returns:
            PipelineResultResponse with overall pipeline results.
        """
        t0 = time.perf_counter()

        try:
            pipeline_id = _new_uuid()
            stages_completed = 0
            assets_registered = 0
            transformations_captured = 0
            edges_created = 0
            validation_result: Optional[str] = None
            report_gen = False
            errors: List[str] = []

            # Delegate to engine if available
            engine_result: Optional[Dict[str, Any]] = None
            if self._pipeline_engine is not None:
                try:
                    engine_result = self._pipeline_engine.run_pipeline(
                        scope=scope,
                        register_assets=register_assets,
                        capture_transformations=capture_transformations,
                        validate=validate,
                        generate_report=generate_report,
                        report_format=report_format,
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    logger.warning("Pipeline engine execution failed: %s", exc)

            if engine_result is not None:
                pipeline_id = engine_result.get("pipeline_id", pipeline_id)
                stages_completed = engine_result.get("stages_completed", 0)
                assets_registered = engine_result.get("assets_registered", 0)
                transformations_captured = engine_result.get("transformations_captured", 0)
                edges_created = engine_result.get("edges_created", 0)
                validation_result = engine_result.get("validation_result")
                report_gen = engine_result.get("report_generated", False)
                errors = engine_result.get("errors", [])
            else:
                # Fallback: run stages individually
                if validate:
                    try:
                        val_resp = self.validate_lineage(scope=scope)
                        validation_result = val_resp.result
                        stages_completed += 1
                    except Exception as exc:
                        errors.append(f"validation: {exc}")

                if generate_report:
                    try:
                        self.generate_report(
                            report_format=report_format,
                            scope=scope,
                        )
                        report_gen = True
                        stages_completed += 1
                    except Exception as exc:
                        errors.append(f"report: {exc}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            response = PipelineResultResponse(
                pipeline_id=pipeline_id,
                stages_completed=stages_completed,
                assets_registered=assets_registered,
                transformations_captured=transformations_captured,
                edges_created=edges_created,
                validation_result=validation_result,
                report_generated=report_gen,
                duration_ms=round(elapsed_ms, 2),
                errors=errors,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._pipeline_results[response.pipeline_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="pipeline_result",
                entity_id=response.pipeline_id,
                action="pipeline_completed",
                data={
                    "scope": scope,
                    "stages_completed": stages_completed,
                    "duration_ms": round(elapsed_ms, 2),
                    "errors": errors,
                },
            )

            # Record metrics
            elapsed = time.perf_counter() - t0
            observe_processing_duration("pipeline", elapsed)

            logger.info(
                "Pipeline %s completed: stages=%d assets=%d txns=%d edges=%d "
                "validation=%s report=%s elapsed=%.1fms",
                response.pipeline_id,
                stages_completed,
                assets_registered,
                transformations_captured,
                edges_created,
                validation_result,
                report_gen,
                elapsed_ms,
            )
            return response

        except Exception as exc:
            logger.error("run_pipeline failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Statistics and health
    # ==================================================================

    def get_health(self) -> Dict[str, Any]:
        """Perform a health check on the data lineage tracker service.

        Returns a dictionary with health status for each engine and
        the overall service.

        Returns:
            Dictionary with health check results including:
            - ``status``: Overall service status (healthy, degraded, unhealthy).
            - ``engines``: Per-engine availability status.
            - ``started``: Whether the service has been started.
            - ``statistics``: Summary statistics.
            - ``provenance_chain_valid``: Whether the provenance chain is intact.
            - ``timestamp``: ISO-8601 UTC timestamp of the check.
        """
        t0 = time.perf_counter()

        engines: Dict[str, str] = {
            "asset_registry": "available" if self._asset_registry_engine is not None else "unavailable",
            "transformation_tracker": "available" if self._transformation_tracker_engine is not None else "unavailable",
            "lineage_graph": "available" if self._lineage_graph_engine is not None else "unavailable",
            "impact_analyzer": "available" if self._impact_analyzer_engine is not None else "unavailable",
            "lineage_validator": "available" if self._lineage_validator_engine is not None else "unavailable",
            "lineage_reporter": "available" if self._lineage_reporter_engine is not None else "unavailable",
            "pipeline": "available" if self._pipeline_engine is not None else "unavailable",
        }

        available_count = sum(
            1 for status in engines.values() if status == "available"
        )
        total_engines = len(engines)

        if available_count == total_engines:
            overall_status = "healthy"
        elif available_count >= 4:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Verify provenance chain
        chain_valid = self.provenance.verify_chain()

        result = {
            "status": overall_status,
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_engines,
            "started": self._started,
            "statistics": {
                "total_assets": self._stats.total_assets,
                "total_transformations": self._stats.total_transformations,
                "total_edges": self._stats.total_edges,
                "total_impact_analyses": self._stats.total_impact_analyses,
                "total_validations": self._stats.total_validations,
                "total_reports": self._stats.total_reports,
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

        elapsed = time.perf_counter() - t0
        observe_processing_duration("health_check", elapsed)

        logger.info(
            "Health check: status=%s engines=%d/%d chain_valid=%s",
            overall_status,
            available_count,
            total_engines,
            chain_valid,
        )
        return result

    def get_statistics(self) -> DataLineageStatisticsResponse:
        """Get aggregate statistics for the data lineage tracker service.

        Returns:
            DataLineageStatisticsResponse with current statistics.
        """
        t0 = time.perf_counter()

        # Enrich from engine statistics where available
        if self._asset_registry_engine is not None:
            try:
                reg_stats = self._asset_registry_engine.get_statistics()
                self._stats.total_assets = reg_stats.get(
                    "total_assets", self._stats.total_assets,
                )
            except (AttributeError, Exception):
                pass

        if self._transformation_tracker_engine is not None:
            try:
                txn_stats = self._transformation_tracker_engine.get_statistics()
                self._stats.total_transformations = txn_stats.get(
                    "total_transformations", self._stats.total_transformations,
                )
            except (AttributeError, Exception):
                pass

        if self._lineage_graph_engine is not None:
            try:
                graph_stats = self._lineage_graph_engine.get_statistics()
                self._stats.total_edges = graph_stats.get(
                    "total_edges", self._stats.total_edges,
                )
                self._stats.max_depth = graph_stats.get(
                    "max_depth", self._stats.max_depth,
                )
                self._stats.orphan_count = graph_stats.get(
                    "orphan_count", self._stats.orphan_count,
                )
            except (AttributeError, Exception):
                pass

        elapsed = time.perf_counter() - t0
        observe_processing_duration("statistics", elapsed)

        logger.debug(
            "Statistics: assets=%d transformations=%d edges=%d "
            "analyses=%d validations=%d reports=%d",
            self._stats.total_assets,
            self._stats.total_transformations,
            self._stats.total_edges,
            self._stats.total_impact_analyses,
            self._stats.total_validations,
            self._stats.total_reports,
        )
        return self._stats

    # ==================================================================
    # Provenance and metrics access
    # ==================================================================

    def get_provenance(self) -> ProvenanceTracker:
        """Get the provenance tracker instance.

        Returns:
            ProvenanceTracker instance used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics as a dictionary.

        Returns:
            Dictionary of metric names to current values.
        """
        return {
            "total_assets": self._stats.total_assets,
            "total_transformations": self._stats.total_transformations,
            "total_edges": self._stats.total_edges,
            "total_impact_analyses": self._stats.total_impact_analyses,
            "total_validations": self._stats.total_validations,
            "total_reports": self._stats.total_reports,
            "avg_depth": self._stats.avg_depth,
            "max_depth": self._stats.max_depth,
            "coverage_score": self._stats.coverage_score,
            "orphan_count": self._stats.orphan_count,
            "provenance_entries": self.provenance.entry_count,
            "provenance_chain_valid": self.provenance.verify_chain(),
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def startup(self) -> None:
        """Start the data lineage tracker service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("DataLineageTrackerService already started; skipping")
            return

        logger.info("DataLineageTrackerService starting up...")
        self._started = True
        set_graph_node_count(0)
        set_graph_edge_count(0)
        logger.info("DataLineageTrackerService startup complete")

    def shutdown(self) -> None:
        """Shutdown the data lineage tracker service and release resources."""
        if not self._started:
            return

        self._started = False
        set_graph_node_count(0)
        set_graph_edge_count(0)
        logger.info("DataLineageTrackerService shut down")

    # ==================================================================
    # Internal helpers: dict -> response model conversion
    # ==================================================================

    def _dict_to_asset_response(
        self,
        rec: Dict[str, Any],
    ) -> AssetResponse:
        """Convert a raw engine dict to AssetResponse.

        Args:
            rec: Dictionary from the AssetRegistryEngine.

        Returns:
            AssetResponse model.
        """
        return AssetResponse(
            asset_id=rec.get("asset_id", rec.get("id", "")),
            qualified_name=rec.get("qualified_name", ""),
            asset_type=rec.get("asset_type", "dataset"),
            display_name=rec.get("display_name", ""),
            owner=rec.get("owner", ""),
            tags=rec.get("tags", {}),
            classification=rec.get("classification", "internal"),
            status=rec.get("status", "active"),
            schema_ref=rec.get("schema_ref"),
            description=rec.get("description", ""),
            metadata=rec.get("metadata", {}),
            created_at=rec.get("created_at", ""),
            updated_at=rec.get("updated_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_transformation_response(
        self,
        rec: Dict[str, Any],
    ) -> TransformationResponse:
        """Convert a raw engine dict to TransformationResponse.

        Args:
            rec: Dictionary from the TransformationTrackerEngine.

        Returns:
            TransformationResponse model.
        """
        return TransformationResponse(
            transformation_id=rec.get("id", rec.get("transformation_id", "")),
            transformation_type=rec.get("transformation_type", "filter"),
            agent_id=rec.get("agent_id", ""),
            pipeline_id=rec.get("pipeline_id", ""),
            execution_id=rec.get("execution_id", ""),
            source_asset_ids=rec.get("source_asset_ids", rec.get("source_assets", [])),
            target_asset_ids=rec.get("target_asset_ids", rec.get("target_assets", [])),
            records_in=rec.get("records_in", 0),
            records_out=rec.get("records_out", 0),
            records_filtered=rec.get("records_filtered", 0),
            records_error=rec.get("records_error", 0),
            duration_ms=rec.get("duration_ms", 0.0),
            started_at=rec.get("started_at", ""),
            completed_at=rec.get("completed_at"),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    def _dict_to_edge_response(
        self,
        rec: Dict[str, Any],
    ) -> EdgeResponse:
        """Convert a raw engine dict to EdgeResponse.

        Args:
            rec: Dictionary from the LineageGraphEngine.

        Returns:
            EdgeResponse model.
        """
        return EdgeResponse(
            edge_id=rec.get("edge_id", rec.get("id", "")),
            source_asset_id=rec.get("source_asset_id", rec.get("source_id", "")),
            target_asset_id=rec.get("target_asset_id", rec.get("target_id", "")),
            transformation_id=rec.get("transformation_id"),
            edge_type=rec.get("edge_type", "dataset_level"),
            source_field=rec.get("source_field"),
            target_field=rec.get("target_field"),
            transformation_logic=rec.get("transformation_logic"),
            confidence=rec.get("confidence", 1.0),
            created_at=rec.get("created_at", ""),
            provenance_hash=rec.get("provenance_hash", ""),
        )

    # ==================================================================
    # Internal helpers: impact analysis
    # ==================================================================

    def _build_impact_response(
        self,
        engine_result: Optional[Dict[str, Any]],
        asset_id: str,
        direction: str,
        max_depth: int,
    ) -> ImpactAnalysisResponse:
        """Build an ImpactAnalysisResponse from engine result or defaults.

        Args:
            engine_result: Optional result from the ImpactAnalyzerEngine.
            asset_id: Root asset identifier.
            direction: Traversal direction.
            max_depth: Maximum traversal depth.

        Returns:
            ImpactAnalysisResponse model.
        """
        if engine_result is not None:
            affected = engine_result.get("affected_assets", engine_result.get("sources", []))
            affected_count = len(affected)
            critical = sum(1 for a in affected if a.get("severity") == "critical")
            high = sum(1 for a in affected if a.get("severity") == "high")
            medium = sum(1 for a in affected if a.get("severity") == "medium")
            low = sum(1 for a in affected if a.get("severity") == "low")
            blast = engine_result.get("blast_radius", 0.0)
            depth = engine_result.get("depth", engine_result.get("max_depth_reached", max_depth))
        else:
            affected = []
            affected_count = 0
            critical = high = medium = low = 0
            blast = 0.0
            depth = 0

        response = ImpactAnalysisResponse(
            root_asset_id=asset_id,
            direction=direction,
            depth=depth,
            affected_assets=affected,
            affected_assets_count=affected_count,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            blast_radius=blast,
            created_at=_utcnow_iso(),
        )
        response.provenance_hash = _compute_hash(response)

        # Store in cache
        self._impact_analyses[response.analysis_id] = response

        return response

    def _highest_severity(self, response: ImpactAnalysisResponse) -> str:
        """Determine the highest severity from an impact analysis response.

        Args:
            response: ImpactAnalysisResponse to inspect.

        Returns:
            Highest severity string (critical, high, medium, low, none).
        """
        if response.critical_count > 0:
            return "critical"
        if response.high_count > 0:
            return "high"
        if response.medium_count > 0:
            return "medium"
        if response.low_count > 0:
            return "low"
        return "none"

    # ==================================================================
    # Internal helpers: filtering
    # ==================================================================

    def _filter_assets(
        self,
        assets: List[AssetResponse],
        asset_type: Optional[str],
        owner: Optional[str],
        classification: Optional[str],
        status: Optional[str],
        tag_key: Optional[str],
        query: Optional[str],
    ) -> List[AssetResponse]:
        """Filter asset response list by multiple criteria.

        Args:
            assets: List of AssetResponse instances.
            asset_type: Exact asset type filter.
            owner: Exact owner filter.
            classification: Exact classification filter.
            status: Exact status filter.
            tag_key: Tag key presence filter.
            query: Substring search in qualified_name and display_name.

        Returns:
            Filtered list of AssetResponse instances.
        """
        result = assets
        if asset_type is not None:
            result = [a for a in result if a.asset_type == asset_type]
        if owner is not None:
            result = [a for a in result if a.owner == owner]
        if classification is not None:
            result = [a for a in result if a.classification == classification]
        if status is not None:
            result = [a for a in result if a.status == status]
        if tag_key is not None:
            result = [a for a in result if tag_key in a.tags]
        if query is not None:
            q = query.lower()
            result = [
                a for a in result
                if q in a.qualified_name.lower() or q in a.display_name.lower()
            ]
        return result


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> DataLineageTrackerService:
    """Get or create the singleton DataLineageTrackerService instance.

    Returns:
        The singleton DataLineageTrackerService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = DataLineageTrackerService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_data_lineage_tracker(
    app: Any,
    config: Optional[DataLineageTrackerConfig] = None,
) -> DataLineageTrackerService:
    """Configure the Data Lineage Tracker Service on a FastAPI application.

    Creates the DataLineageTrackerService, stores it in app.state, mounts
    the data lineage API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional data lineage tracker config.

    Returns:
        DataLineageTrackerService instance.
    """
    global _singleton_instance

    service = DataLineageTrackerService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.data_lineage_tracker_service = service

    # Mount data lineage tracker API router
    router = get_router()
    if router is not None:
        app.include_router(router)
        logger.info("Data lineage tracker API router mounted")
    else:
        logger.warning(
            "Data lineage tracker router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Data lineage tracker service configured on app")
    return service


def get_data_lineage_tracker() -> DataLineageTrackerService:
    """Get the singleton DataLineageTrackerService instance.

    Returns:
        DataLineageTrackerService singleton instance.

    Raises:
        RuntimeError: If data lineage tracker service not configured.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = DataLineageTrackerService()
    return _singleton_instance


def get_router(service: Optional[DataLineageTrackerService] = None) -> Any:
    """Get the data lineage tracker API router.

    Creates a FastAPI APIRouter with all 20 endpoints for the Data
    Lineage Tracker service at prefix ``/api/v1/data-lineage``.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from fastapi import APIRouter, HTTPException, Query
        from fastapi.responses import JSONResponse
    except ImportError:
        return None

    router = APIRouter(
        prefix="/api/v1/data-lineage",
        tags=["data-lineage"],
    )

    def _svc() -> DataLineageTrackerService:
        """Get the singleton service for route handlers."""
        return get_data_lineage_tracker()

    # ------------------------------------------------------------------
    # 1. POST /assets - Register a new data asset
    # ------------------------------------------------------------------
    @router.post("/assets", response_model=AssetResponse, status_code=201)
    async def post_register_asset(
        request: Dict[str, Any],
    ) -> AssetResponse:
        """Register a new data asset in the lineage graph."""
        try:
            return _svc().register_asset(
                qualified_name=request.get("qualified_name", ""),
                asset_type=request.get("asset_type", "dataset"),
                display_name=request.get("display_name", ""),
                owner=request.get("owner", ""),
                tags=request.get("tags"),
                classification=request.get("classification", "internal"),
                schema_ref=request.get("schema_ref"),
                description=request.get("description", ""),
                metadata=request.get("metadata"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. GET /assets/{asset_id} - Get a data asset by ID
    # ------------------------------------------------------------------
    @router.get("/assets/{asset_id}", response_model=AssetResponse)
    async def get_asset_by_id(asset_id: str) -> AssetResponse:
        """Get a data asset by its unique identifier."""
        result = _svc().get_asset(asset_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return result

    # ------------------------------------------------------------------
    # 3. PUT /assets/{asset_id} - Update a data asset
    # ------------------------------------------------------------------
    @router.put("/assets/{asset_id}", response_model=AssetResponse)
    async def put_update_asset(
        asset_id: str,
        request: Dict[str, Any],
    ) -> AssetResponse:
        """Update mutable fields of an existing data asset."""
        result = _svc().update_asset(
            asset_id=asset_id,
            display_name=request.get("display_name"),
            owner=request.get("owner"),
            tags=request.get("tags"),
            classification=request.get("classification"),
            status=request.get("status"),
            description=request.get("description"),
            metadata=request.get("metadata"),
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return result

    # ------------------------------------------------------------------
    # 4. DELETE /assets/{asset_id} - Soft-delete a data asset
    # ------------------------------------------------------------------
    @router.delete("/assets/{asset_id}", status_code=204)
    async def delete_asset_by_id(asset_id: str) -> None:
        """Soft-delete a data asset by archiving it."""
        deleted = _svc().delete_asset(asset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Asset not found")

    # ------------------------------------------------------------------
    # 5. GET /assets - Search data assets
    # ------------------------------------------------------------------
    @router.get("/assets", response_model=List[AssetResponse])
    async def get_search_assets(
        asset_type: Optional[str] = Query(None),
        owner: Optional[str] = Query(None),
        classification: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        tag_key: Optional[str] = Query(None),
        query: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[AssetResponse]:
        """Search data assets with optional filtering and pagination."""
        return _svc().search_assets(
            asset_type=asset_type,
            owner=owner,
            classification=classification,
            status=status,
            tag_key=tag_key,
            query=query,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 6. POST /transformations - Record a transformation event
    # ------------------------------------------------------------------
    @router.post(
        "/transformations",
        response_model=TransformationResponse,
        status_code=201,
    )
    async def post_record_transformation(
        request: Dict[str, Any],
    ) -> TransformationResponse:
        """Record a data transformation event in the lineage graph."""
        try:
            return _svc().record_transformation(
                transformation_type=request.get("transformation_type", ""),
                agent_id=request.get("agent_id", ""),
                pipeline_id=request.get("pipeline_id", ""),
                execution_id=request.get("execution_id", ""),
                source_asset_ids=request.get("source_asset_ids"),
                target_asset_ids=request.get("target_asset_ids"),
                description=request.get("description", ""),
                parameters=request.get("parameters"),
                records_in=request.get("records_in", 0),
                records_out=request.get("records_out", 0),
                records_filtered=request.get("records_filtered", 0),
                records_error=request.get("records_error", 0),
                duration_ms=request.get("duration_ms", 0.0),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. GET /transformations - List transformation events
    # ------------------------------------------------------------------
    @router.get("/transformations", response_model=List[TransformationResponse])
    async def get_list_transformations(
        agent_id: Optional[str] = Query(None),
        pipeline_id: Optional[str] = Query(None),
        transformation_type: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[TransformationResponse]:
        """List transformation events with optional filtering."""
        return _svc().list_transformations(
            agent_id=agent_id,
            pipeline_id=pipeline_id,
            transformation_type=transformation_type,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 8. POST /edges - Create a lineage edge
    # ------------------------------------------------------------------
    @router.post("/edges", response_model=EdgeResponse, status_code=201)
    async def post_create_edge(
        request: Dict[str, Any],
    ) -> EdgeResponse:
        """Create a lineage edge between two data assets."""
        try:
            return _svc().create_edge(
                source_asset_id=request.get("source_asset_id", ""),
                target_asset_id=request.get("target_asset_id", ""),
                transformation_id=request.get("transformation_id"),
                edge_type=request.get("edge_type", "dataset_level"),
                source_field=request.get("source_field"),
                target_field=request.get("target_field"),
                transformation_logic=request.get("transformation_logic"),
                confidence=request.get("confidence", 1.0),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. GET /edges - List lineage edges
    # ------------------------------------------------------------------
    @router.get("/edges", response_model=List[EdgeResponse])
    async def get_list_edges(
        source_asset_id: Optional[str] = Query(None),
        target_asset_id: Optional[str] = Query(None),
        edge_type: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> List[EdgeResponse]:
        """List lineage edges with optional filtering."""
        return _svc().list_edges(
            source_asset_id=source_asset_id,
            target_asset_id=target_asset_id,
            edge_type=edge_type,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 10. GET /graph - Get full lineage graph snapshot
    # ------------------------------------------------------------------
    @router.get("/graph", response_model=GraphResponse)
    async def get_graph_snapshot() -> GraphResponse:
        """Get the full lineage graph snapshot."""
        return _svc().get_graph()

    # ------------------------------------------------------------------
    # 11. GET /graph/subgraph/{asset_id} - Extract subgraph
    # ------------------------------------------------------------------
    @router.get("/graph/subgraph/{asset_id}", response_model=SubgraphResponse)
    async def get_subgraph_for_asset(
        asset_id: str,
        depth: int = Query(10, ge=1, le=50),
        direction: str = Query("both"),
    ) -> SubgraphResponse:
        """Extract a subgraph around a root asset."""
        try:
            return _svc().get_subgraph(
                root_asset_id=asset_id,
                depth=depth,
                direction=direction,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. GET /impact/backward/{asset_id} - Backward lineage traversal
    # ------------------------------------------------------------------
    @router.get(
        "/impact/backward/{asset_id}",
        response_model=ImpactAnalysisResponse,
    )
    async def get_backward_analysis(
        asset_id: str,
        max_depth: int = Query(10, ge=1, le=50),
    ) -> ImpactAnalysisResponse:
        """Perform backward lineage traversal (provenance)."""
        try:
            return _svc().analyze_backward(asset_id, max_depth=max_depth)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. GET /impact/forward/{asset_id} - Forward impact analysis
    # ------------------------------------------------------------------
    @router.get(
        "/impact/forward/{asset_id}",
        response_model=ImpactAnalysisResponse,
    )
    async def get_forward_analysis(
        asset_id: str,
        max_depth: int = Query(10, ge=1, le=50),
    ) -> ImpactAnalysisResponse:
        """Perform forward impact analysis."""
        try:
            return _svc().analyze_forward(asset_id, max_depth=max_depth)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. POST /impact - Run impact analysis (configurable direction)
    # ------------------------------------------------------------------
    @router.post("/impact", response_model=ImpactAnalysisResponse)
    async def post_run_impact_analysis(
        request: Dict[str, Any],
    ) -> ImpactAnalysisResponse:
        """Run impact analysis in the specified direction."""
        try:
            return _svc().run_impact_analysis(
                asset_id=request.get("asset_id", ""),
                direction=request.get("direction", "forward"),
                max_depth=request.get("max_depth", 10),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. POST /validate - Run lineage validation
    # ------------------------------------------------------------------
    @router.post("/validate", response_model=ValidationResponse)
    async def post_validate_lineage(
        request: Dict[str, Any],
    ) -> ValidationResponse:
        """Run lineage graph validation and health check."""
        return _svc().validate_lineage(
            scope=request.get("scope", "full"),
            include_freshness=request.get("include_freshness", True),
            include_coverage=request.get("include_coverage", True),
        )

    # ------------------------------------------------------------------
    # 16. GET /validate/{validation_id} - Get validation result
    # ------------------------------------------------------------------
    @router.get(
        "/validate/{validation_id}",
        response_model=ValidationResponse,
    )
    async def get_validation_result(
        validation_id: str,
    ) -> ValidationResponse:
        """Get a validation result by its identifier."""
        result = _svc().get_validation(validation_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Validation result not found",
            )
        return result

    # ------------------------------------------------------------------
    # 17. POST /reports - Generate a lineage report
    # ------------------------------------------------------------------
    @router.post("/reports", response_model=ReportResponse, status_code=201)
    async def post_generate_report(
        request: Dict[str, Any],
    ) -> ReportResponse:
        """Generate a lineage report in the specified format."""
        return _svc().generate_report(
            report_type=request.get("report_type", "custom"),
            report_format=request.get("format", "json"),
            scope=request.get("scope", "full"),
            parameters=request.get("parameters"),
            max_depth=request.get("max_depth", 10),
        )

    # ------------------------------------------------------------------
    # 18. POST /pipeline - Run end-to-end lineage pipeline
    # ------------------------------------------------------------------
    @router.post("/pipeline", response_model=PipelineResultResponse)
    async def post_run_pipeline(
        request: Dict[str, Any],
    ) -> PipelineResultResponse:
        """Run the full end-to-end data lineage pipeline."""
        return _svc().run_pipeline(
            scope=request.get("scope", "full"),
            register_assets=request.get("register_assets", True),
            capture_transformations=request.get("capture_transformations", True),
            validate=request.get("validate", True),
            generate_report=request.get("generate_report", False),
            report_format=request.get("report_format", "json"),
        )

    # ------------------------------------------------------------------
    # 19. GET /health - Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def get_health_check() -> Dict[str, Any]:
        """Perform a health check on the data lineage tracker service."""
        return _svc().get_health()

    # ------------------------------------------------------------------
    # 20. GET /statistics - Aggregate statistics
    # ------------------------------------------------------------------
    @router.get("/statistics", response_model=DataLineageStatisticsResponse)
    async def get_service_statistics() -> DataLineageStatisticsResponse:
        """Get aggregate statistics for the data lineage tracker service."""
        return _svc().get_statistics()

    return router


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service class
    "DataLineageTrackerService",
    # FastAPI integration
    "configure_data_lineage_tracker",
    "get_data_lineage_tracker",
    "get_router",
    # Response models
    "AssetResponse",
    "TransformationResponse",
    "EdgeResponse",
    "GraphResponse",
    "SubgraphResponse",
    "ImpactAnalysisResponse",
    "ValidationResponse",
    "ReportResponse",
    "PipelineResultResponse",
    "DataLineageStatisticsResponse",
]
