# -*- coding: utf-8 -*-
"""
Plot Boundary Manager Agent - AGENT-EUDR-006

Geospatial plot boundary management engine for EU Deforestation
Regulation (EUDR) Articles 9, 10, and 31 compliance. Manages plot
boundary polygons with OGC-compliant validation, geodetic area
calculation using the Karney algorithm on the WGS84 ellipsoid,
spatial overlap detection with R-tree indexing, immutable boundary
versioning with 5-year retention, polygon simplification preserving
topological relationships, split/merge operations with genealogy
tracking and area conservation verification, and multi-format export
for regulatory submission.

This package contains:
    Foundational modules:
        - models: Pydantic v2 data models for geometry types, coordinate
          reference systems, boundary validation, area calculation,
          overlap detection, versioning, simplification, split/merge,
          and compliance export
        - config: PlotBoundaryConfig with GL_EUDR_PBM_ env var support
        - provenance: SHA-256 chain-hashed audit trail tracking
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_pbm_ prefix)

    Engine modules:
        - polygon_manager: Polygon CRUD with CRS reprojection
        - boundary_validator: OGC validation and auto-repair
        - area_calculator: Karney ellipsoidal area computation
        - overlap_detector: R-tree spatial overlap scanning
        - boundary_versioner: Immutable version management
        - simplification_engine: Polygon simplification/generalization
        - split_merge_engine: Split/merge with genealogy tracking
        - compliance_reporter: Multi-format export and compliance reports

PRD: PRD-AGENT-EUDR-006
Agent ID: GL-EUDR-PBM-006
Regulation: EU 2023/1115 (EUDR) Article 9, Article 10, Article 31
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.plot_boundary import (
    ...     CreateBoundaryRequest,
    ...     PlotBoundary,
    ...     GeometryType,
    ...     ThresholdClassification,
    ...     OverlapSeverity,
    ...     ExportFormat,
    ... )
    >>> request = CreateBoundaryRequest(
    ...     exterior_ring_coords=[
    ...         [-3.4, -62.2], [-3.4, -62.1],
    ...         [-3.5, -62.1], [-3.5, -62.2], [-3.4, -62.2],
    ...     ],
    ...     commodity="soya",
    ...     country_iso="BR",
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# ---- Foundational: config ----
from greenlang.agents.eudr.plot_boundary.config import (
    PlotBoundaryConfig,
    get_config,
    set_config,
    reset_config,
)

# ---- Foundational: models ----
from greenlang.agents.eudr.plot_boundary.models import (
    # Constants
    VERSION,
    EUDR_AREA_THRESHOLD_HA,
    MAX_BATCH_SIZE,
    DEFAULT_COORDINATE_PRECISION,
    WGS84_SEMI_MAJOR_AXIS,
    WGS84_FLATTENING,
    MIN_POLYGON_VERTICES,
    # Re-exported from greenlang.agents.data.eudr_traceability.models
    EUDRCommodity,
    # Enumerations
    GeometryType,
    CoordinateReferenceSystem,
    ValidationErrorType,
    RepairStrategy,
    OverlapSeverity,
    OverlapResolution,
    VersionChangeReason,
    SimplificationMethod,
    ExportFormat,
    ThresholdClassification,
    CompactnessIndex,
    BatchStatus,
    # Core models
    Coordinate,
    BoundingBox,
    Ring,
    PlotBoundary,
    ValidationError,
    ValidationResult,
    AreaResult,
    OverlapRecord,
    BoundaryVersion,
    SimplificationResult,
    SplitResult,
    MergeResult,
    ExportResult,
    BatchJob,
    # Request models
    CreateBoundaryRequest,
    UpdateBoundaryRequest,
    ValidateRequest,
    RepairRequest,
    AreaCalculationRequest,
    OverlapDetectionRequest,
    SimplifyRequest,
    SplitRequest,
    MergeRequest,
    ExportRequest,
    BatchBoundaryRequest,
    # Response models
    BoundaryResponse,
    ValidationResponse,
    AreaResponse,
    OverlapResponse,
    VersionResponse,
    SimplificationResponse,
    SplitMergeResponse,
    ExportResponse,
)

# ---- Foundational: provenance ----
from greenlang.agents.eudr.plot_boundary.provenance import (
    ProvenanceRecord,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---- Foundational: metrics ----
from greenlang.agents.eudr.plot_boundary.metrics import (
    PROMETHEUS_AVAILABLE,
    record_boundary_created,
    record_boundary_updated,
    record_validation,
    record_validation_error,
    record_repair,
    record_area_calculation,
    record_overlap_detected,
    record_overlap_scan,
    record_version_created,
    record_simplification,
    record_split,
    record_merge,
    record_export,
    record_batch_job,
    record_operation_duration,
    record_vertex_count,
    record_area_hectares,
    record_api_error,
)

# ---- Engine modules (optional imports for forward compatibility) ----
try:
    from greenlang.agents.eudr.plot_boundary.polygon_manager import (
        PolygonManager,
    )
except ImportError:
    PolygonManager = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.boundary_validator import (
        BoundaryValidator,
    )
except ImportError:
    BoundaryValidator = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.area_calculator import (
        AreaCalculator,
    )
except ImportError:
    AreaCalculator = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.overlap_detector import (
        OverlapDetector,
    )
except ImportError:
    OverlapDetector = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.boundary_versioner import (
        BoundaryVersioner,
    )
except ImportError:
    BoundaryVersioner = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.simplification_engine import (
        SimplificationEngine,
    )
except ImportError:
    SimplificationEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.split_merge_engine import (
        SplitMergeEngine,
    )
except ImportError:
    SplitMergeEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.eudr.plot_boundary.compliance_reporter import (
        ComplianceReporter,
    )
except ImportError:
    ComplianceReporter = None  # type: ignore[assignment, misc]

# ---- Service (optional import for forward compatibility) ----
try:
    from greenlang.agents.eudr.plot_boundary.setup import (
        PlotBoundaryService,
    )
except ImportError:
    PlotBoundaryService = None  # type: ignore[assignment, misc]

# ---- Version ----
__version__ = "1.0.0"

__all__ = [
    # -- Version --
    "__version__",
    # -- Config --
    "PlotBoundaryConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "VERSION",
    "EUDR_AREA_THRESHOLD_HA",
    "MAX_BATCH_SIZE",
    "DEFAULT_COORDINATE_PRECISION",
    "WGS84_SEMI_MAJOR_AXIS",
    "WGS84_FLATTENING",
    "MIN_POLYGON_VERTICES",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "GeometryType",
    "CoordinateReferenceSystem",
    "ValidationErrorType",
    "RepairStrategy",
    "OverlapSeverity",
    "OverlapResolution",
    "VersionChangeReason",
    "SimplificationMethod",
    "ExportFormat",
    "ThresholdClassification",
    "CompactnessIndex",
    "BatchStatus",
    # -- Core Models --
    "Coordinate",
    "BoundingBox",
    "Ring",
    "PlotBoundary",
    "ValidationError",
    "ValidationResult",
    "AreaResult",
    "OverlapRecord",
    "BoundaryVersion",
    "SimplificationResult",
    "SplitResult",
    "MergeResult",
    "ExportResult",
    "BatchJob",
    # -- Request Models --
    "CreateBoundaryRequest",
    "UpdateBoundaryRequest",
    "ValidateRequest",
    "RepairRequest",
    "AreaCalculationRequest",
    "OverlapDetectionRequest",
    "SimplifyRequest",
    "SplitRequest",
    "MergeRequest",
    "ExportRequest",
    "BatchBoundaryRequest",
    # -- Response Models --
    "BoundaryResponse",
    "ValidationResponse",
    "AreaResponse",
    "OverlapResponse",
    "VersionResponse",
    "SimplificationResponse",
    "SplitMergeResponse",
    "ExportResponse",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_boundary_created",
    "record_boundary_updated",
    "record_validation",
    "record_validation_error",
    "record_repair",
    "record_area_calculation",
    "record_overlap_detected",
    "record_overlap_scan",
    "record_version_created",
    "record_simplification",
    "record_split",
    "record_merge",
    "record_export",
    "record_batch_job",
    "record_operation_duration",
    "record_vertex_count",
    "record_area_hectares",
    "record_api_error",
    # -- Engine Classes (optional) --
    "PolygonManager",
    "BoundaryValidator",
    "AreaCalculator",
    "OverlapDetector",
    "BoundaryVersioner",
    "SimplificationEngine",
    "SplitMergeEngine",
    "ComplianceReporter",
    # -- Service (optional) --
    "PlotBoundaryService",
]
