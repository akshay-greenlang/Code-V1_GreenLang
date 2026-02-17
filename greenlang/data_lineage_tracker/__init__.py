# -*- coding: utf-8 -*-
"""
GL-DATA-X-021: GreenLang Data Lineage Tracker SDK
===================================================

This package provides data asset registry management, transformation tracking,
lineage graph construction, impact analysis, lineage validation, lineage
reporting, and end-to-end pipeline orchestration SDK for the GreenLang
framework. It supports:

- Asset registry with multi-type support (datasets, tables, files, APIs,
  models, reports) and namespace isolation
- Transformation tracking with input/output capture, SQL/code provenance,
  and parameter recording for full reproducibility
- Lineage graph construction with directed acyclic graph (DAG) representation,
  upstream/downstream traversal, and subgraph extraction
- Impact analysis for change propagation assessment across downstream
  consumers with blast radius estimation and risk scoring
- Lineage validation for completeness, consistency, orphan detection,
  cycle detection, and regulatory compliance checks
- Lineage reporting with HTML, JSON, Markdown, and PDF export formats
  for audit trails and stakeholder communication
- End-to-end pipeline orchestration chaining all 6 engines with
  configurable stages and short-circuit on failure
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics with gl_dlt_ prefix for observability
- FastAPI REST API with 20 endpoints at /api/v1/data-lineage-tracker
- Thread-safe configuration with GL_DLT_ env prefix

Key Components:
    - config: DataLineageTrackerConfig with GL_DLT_ env prefix
    - asset_registry: Data asset registration and lookup engine
    - transformation_tracker: Transformation capture and tracking engine
    - lineage_graph: Lineage graph construction and traversal engine
    - impact_analyzer: Change impact analysis and blast radius engine
    - lineage_validator: Lineage completeness and consistency validation engine
    - lineage_reporter: Lineage report generation engine
    - lineage_tracker_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_dlt_ prefix
    - api: FastAPI HTTP service with 20 endpoints
    - setup: DataLineageTrackerService facade

Example:
    >>> from greenlang.data_lineage_tracker import DataLineageTrackerService
    >>> service = DataLineageTrackerService()
    >>> result = service.register_asset(
    ...     name="emissions_raw",
    ...     asset_type="dataset",
    ...     schema={"columns": ["co2e", "source", "timestamp"]},
    ...     namespace="emissions",
    ... )
    >>> print(result.asset_id, result.status)
    emissions_raw active

Agent ID: GL-DATA-X-021
Agent Name: Data Lineage Tracker
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-021"
__agent_name__ = "Data Lineage Tracker"

# SDK availability flag
DATA_LINEAGE_TRACKER_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.data_lineage_tracker.config import (
    DataLineageTrackerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.data_lineage_tracker.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    dlt_assets_registered_total,
    dlt_transformations_captured_total,
    dlt_edges_created_total,
    dlt_impact_analyses_total,
    dlt_validations_total,
    dlt_reports_generated_total,
    dlt_change_events_total,
    dlt_quality_scores_computed_total,
    dlt_graph_traversal_duration_seconds,
    dlt_processing_duration_seconds,
    dlt_graph_node_count,
    dlt_graph_edge_count,
    # Helper functions
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

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
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

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.data_lineage_tracker.setup import (
    DataLineageTrackerService,
    configure_data_lineage_tracker,
    get_data_lineage_tracker,
    get_router,
    # Models
    AssetResponse,
    TransformationResponse,
    EdgeResponse,
    GraphResponse,
    SubgraphResponse,
    ImpactAnalysisResponse,
    ValidationResponse,
    ReportResponse,
    PipelineResultResponse,
    DataLineageStatisticsResponse,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "DATA_LINEAGE_TRACKER_SDK_AVAILABLE",
    # Configuration
    "DataLineageTrackerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "dlt_assets_registered_total",
    "dlt_transformations_captured_total",
    "dlt_edges_created_total",
    "dlt_impact_analyses_total",
    "dlt_validations_total",
    "dlt_reports_generated_total",
    "dlt_change_events_total",
    "dlt_quality_scores_computed_total",
    "dlt_graph_traversal_duration_seconds",
    "dlt_processing_duration_seconds",
    "dlt_graph_node_count",
    "dlt_graph_edge_count",
    # Metric helper functions
    "record_asset_registered",
    "record_transformation_captured",
    "record_edge_created",
    "record_impact_analysis",
    "record_validation",
    "record_report_generated",
    "record_change_event",
    "record_quality_score",
    "observe_graph_traversal_duration",
    "observe_processing_duration",
    "set_graph_node_count",
    "set_graph_edge_count",
    # Core engines (Layer 2)
    "AssetRegistryEngine",
    "TransformationTrackerEngine",
    "LineageGraphEngine",
    "ImpactAnalyzerEngine",
    "LineageValidatorEngine",
    "LineageReporterEngine",
    "LineageTrackerPipelineEngine",
    # Service setup facade
    "DataLineageTrackerService",
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
