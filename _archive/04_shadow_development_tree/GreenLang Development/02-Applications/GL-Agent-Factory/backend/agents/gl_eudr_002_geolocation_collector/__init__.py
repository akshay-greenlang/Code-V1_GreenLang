"""
GL-EUDR-002: Geolocation Collector Agent

Collects, validates, and manages production plot geolocation data
for EUDR compliance. This agent ensures all origin locations meet
EUDR's strict requirements (WGS-84, 6 decimal precision).

Features:
- Multi-channel data collection (GPS, web form, bulk upload, API)
- Deterministic coordinate validation (zero hallucination)
- Plot geometry management (Point and Polygon)
- Data enrichment (geocoding, admin regions, biomes)
- PostGIS integration for spatial operations

Regulatory Reference:
    EU Regulation 2023/1115 (EUDR) - Article 9
    Enforcement Date: December 30, 2024 (Large Operators)
    SME Enforcement Date: June 30, 2025

Example:
    >>> from backend.agents.gl_eudr_002_geolocation_collector import (
    ...     GeolocationCollectorAgent,
    ...     GeolocationInput,
    ...     PointCoordinates,
    ...     CommodityType,
    ...     OperationType,
    ... )
    >>> agent = GeolocationCollectorAgent()
    >>> result = agent.run(GeolocationInput(
    ...     operation=OperationType.VALIDATE_COORDINATES,
    ...     coordinates=PointCoordinates(latitude=-4.123456, longitude=102.654321),
    ...     country_code="ID",
    ...     commodity=CommodityType.PALM_OIL
    ... ))
    >>> print(f"Valid: {result.validation_result.valid}")
"""

from .agent import (
    # Main Agent
    GeolocationCollectorAgent,

    # Input/Output Models
    GeolocationInput,
    GeolocationOutput,

    # Core Enums
    CommodityType,
    GeometryType,
    ValidationStatus,
    ValidationSeverity,
    CollectionMethod,
    OperationType,
    BulkUploadFormat,
    BulkJobStatus,
    ErrorCode,
    ERROR_SEVERITY,

    # Coordinate Models
    PointCoordinates,
    PolygonCoordinates,

    # Validation Models
    ValidationError,
    ValidationResult,
    StageResult,

    # Plot Models
    PlotSubmission,
    Plot,
    PlotValidationHistory,

    # Bulk Upload Models
    BulkUploadJob,
    BulkUploadResult,

    # Validation Engine
    GeolocationValidator,

    # Pack Spec
    PACK_SPEC,
)

from .spatial import (
    # Main Spatial Service
    SpatialValidationService,

    # Individual Services
    CountryBoundaryService,
    WaterBodyService,
    ProtectedAreaService,
    UrbanAreaService,

    # Spatial Models
    SpatialFeature,
    SpatialQueryResult,
    BoundingBox,
    SpatialIndex,

    # Data Loading
    SpatialDataSource,
    BoundaryLevel,
    GeoJSONLoader,

    # Utilities
    normalize_longitude,
    crosses_dateline,
    split_dateline_polygon,
)

from .database import (
    # Configuration
    DatabaseConfig,

    # Session Management
    DatabaseSession,

    # Repositories
    PlotRepository,
    ValidationHistoryRepository,
    BulkJobRepository,
)

from .llm_service import (
    # Service
    GeolocationLLMService,
    LLMProvider,

    # Data Classes
    ParsedAddress,
    ValidationExplanation,
    LocationDescription,
    CoordinateFormatHelp,
)

from .bulk_upload import (
    # Processor
    BulkUploadProcessor,
    BulkJobQueue,

    # Parsers
    BulkFileParser,
    CSVParser,
    GeoJSONParser,
    KMLParser,

    # Data Classes
    ParsedRecord,
    ValidationBatch,
    ProcessingProgress,
    ProcessingResult,

    # Utilities
    generate_processing_report,
)

__all__ = [
    # Main Agent
    "GeolocationCollectorAgent",

    # Input/Output Models
    "GeolocationInput",
    "GeolocationOutput",

    # Core Enums
    "CommodityType",
    "GeometryType",
    "ValidationStatus",
    "ValidationSeverity",
    "CollectionMethod",
    "OperationType",
    "BulkUploadFormat",
    "BulkJobStatus",
    "ErrorCode",
    "ERROR_SEVERITY",

    # Coordinate Models
    "PointCoordinates",
    "PolygonCoordinates",

    # Validation Models
    "ValidationError",
    "ValidationResult",
    "StageResult",

    # Plot Models
    "PlotSubmission",
    "Plot",
    "PlotValidationHistory",

    # Bulk Upload Models
    "BulkUploadJob",
    "BulkUploadResult",

    # Validation Engine
    "GeolocationValidator",

    # Pack Spec
    "PACK_SPEC",

    # Spatial Validation Services
    "SpatialValidationService",
    "CountryBoundaryService",
    "WaterBodyService",
    "ProtectedAreaService",
    "UrbanAreaService",

    # Spatial Models
    "SpatialFeature",
    "SpatialQueryResult",
    "BoundingBox",
    "SpatialIndex",

    # Spatial Data Loading
    "SpatialDataSource",
    "BoundaryLevel",
    "GeoJSONLoader",

    # Spatial Utilities
    "normalize_longitude",
    "crosses_dateline",
    "split_dateline_polygon",

    # Database
    "DatabaseConfig",
    "DatabaseSession",
    "PlotRepository",
    "ValidationHistoryRepository",
    "BulkJobRepository",

    # LLM Service
    "GeolocationLLMService",
    "LLMProvider",
    "ParsedAddress",
    "ValidationExplanation",
    "LocationDescription",
    "CoordinateFormatHelp",

    # Bulk Upload
    "BulkUploadProcessor",
    "BulkJobQueue",
    "BulkFileParser",
    "CSVParser",
    "GeoJSONParser",
    "KMLParser",
    "ParsedRecord",
    "ValidationBatch",
    "ProcessingProgress",
    "ProcessingResult",
    "generate_processing_report",
]

__version__ = "1.0.0"
__agent_id__ = "eudr/geolocation_collector_v1"
