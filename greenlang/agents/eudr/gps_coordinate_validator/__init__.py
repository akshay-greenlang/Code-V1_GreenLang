# -*- coding: utf-8 -*-
"""
GPS Coordinate Validator - AGENT-EUDR-007

Production-grade GPS coordinate validation, parsing, datum transformation,
precision assessment, spatial plausibility checking, reverse geocoding,
accuracy scoring, and EUDR Article 9 compliance reporting for the EU
Deforestation Regulation (EUDR).

This package provides an 8-engine validation pipeline for GPS coordinate
data quality assurance:

    Engine 1 - CoordinateParser:
        Multi-format coordinate parsing supporting DD, DMS, DDM, UTM,
        MGRS, signed DD, and DD with hemisphere suffix formats.

    Engine 2 - DatumTransformer:
        Geodetic datum transformation supporting 30+ datums via Helmert
        7-parameter and Molodensky 3-parameter methods. Transforms all
        input coordinates to WGS84 (EPSG:4326) canonical CRS.

    Engine 3 - PrecisionAnalyzer:
        Precision assessment including decimal place counting, ground
        resolution calculation, EUDR adequacy checking, truncation
        detection, and artificial rounding detection.

    Engine 4 - FormatValidator:
        Format validation and error detection including WGS84 range
        checking, lat/lon swap detection, sign error detection, null
        island detection, NaN/Inf/null value detection, duplicate
        detection, and format error classification.

    Engine 5 - SpatialPlausibilityChecker:
        Spatial plausibility checking including ocean detection, country
        boundary verification, commodity production zone assessment,
        elevation plausibility checking, urban area detection, and
        protected area intersection analysis.

    Engine 6 - ReverseGeocoder:
        Reverse geocoding providing country identification, administrative
        region resolution, nearest place name lookup, land use
        classification, coast distance calculation, commodity zone
        identification, and elevation data retrieval.

    Engine 7 - AccuracyAssessor:
        Composite accuracy scoring across four quality dimensions
        (precision, plausibility, consistency, source quality) with
        configurable weights and accuracy tier classification (Gold,
        Silver, Bronze, Unverified).

    Engine 8 - ComplianceReporter:
        EUDR Article 9 compliance reporting in JSON, PDF, CSV, and
        EUDR XML formats with compliance certificate issuance and
        provenance-tracked audit trails.

Foundational modules:
    - models: Pydantic v2 data models with 14 enumerations, 7 core models,
      4 result models, 8 request models, and 8 response models
    - config: GPSCoordinateValidatorConfig with GL_EUDR_GCV_ env var support
    - provenance: SHA-256 chain-hashed audit trail tracking with 10 entity
      types and 11 actions
    - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_gcv_ prefix)

PRD: PRD-AGENT-EUDR-007
Agent ID: GL-EUDR-GCV-007
Regulation: EU 2023/1115 (EUDR) Article 9
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.gps_coordinate_validator import (
    ...     RawCoordinate,
    ...     CoordinateFormat,
    ...     GeodeticDatum,
    ...     SourceType,
    ...     GPSCoordinateValidatorConfig,
    ...     get_config,
    ... )
    >>> raw = RawCoordinate(
    ...     raw_input="-3.4653, -62.2159",
    ...     country_iso="BR",
    ...     commodity="soya",
    ...     source_type=SourceType.MOBILE_GPS,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-GCV-007"

# ---- Foundational: config ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.config import (
        GPSCoordinateValidatorConfig,
        get_config,
        set_config,
        reset_config,
    )
except ImportError:
    GPSCoordinateValidatorConfig = None  # type: ignore[assignment,misc]
    get_config = None  # type: ignore[assignment]
    set_config = None  # type: ignore[assignment]
    reset_config = None  # type: ignore[assignment]

# ---- Foundational: models ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.models import (
        # Constants
        VERSION,
        EUDR_DEFORESTATION_CUTOFF,
        MAX_BATCH_SIZE,
        DEFAULT_QUALITY_WEIGHTS,
        ACCURACY_TIER_THRESHOLDS,
        EUDR_MIN_DECIMAL_PLACES,
        WGS84_SEMI_MAJOR_AXIS,
        WGS84_INV_FLATTENING,
        WGS84_FLATTENING,
        # Re-exported commodity enum
        EUDRCommodity,
        # Enumerations
        CoordinateFormat,
        GeodeticDatum,
        PrecisionLevel,
        ValidationErrorType,
        PlausibilityCheckResult,
        AccuracyTier,
        CorrectionType,
        ReportFormat,
        ComplianceStatus,
        SourceType,
        LandUseContext,
        BatchStatus,
        ElevationSource,
        HemisphereIndicator,
        # Core models
        RawCoordinate,
        ParsedCoordinate,
        NormalizedCoordinate,
        CoordinateValidationError,
        ValidationResult,
        PrecisionResult,
        DatumTransformResult,
        # Result models
        PlausibilityResult,
        ReverseGeocodeResult,
        AccuracyScore,
        ComplianceCertificate,
        # Request models
        ParseCoordinateRequest,
        ValidateCoordinateRequest,
        TransformDatumRequest,
        AnalyzePrecisionRequest,
        CheckPlausibilityRequest,
        ReverseGeocodeRequest,
        AssessAccuracyRequest,
        GenerateReportRequest,
        # Response models
        ParseCoordinateResponse,
        ValidateCoordinateResponse,
        TransformDatumResponse,
        AnalyzePrecisionResponse,
        CheckPlausibilityResponse,
        ReverseGeocodeResponse,
        AssessAccuracyResponse,
        BatchValidationResult,
    )
except ImportError:
    pass

# ---- Foundational: provenance ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.provenance import (
        ProvenanceRecord,
        ProvenanceTracker,
        VALID_ENTITY_TYPES,
        VALID_ACTIONS,
        get_provenance_tracker,
        set_provenance_tracker,
        reset_provenance_tracker,
    )
except ImportError:
    ProvenanceRecord = None  # type: ignore[assignment,misc]
    ProvenanceTracker = None  # type: ignore[assignment,misc]
    VALID_ENTITY_TYPES = frozenset()  # type: ignore[assignment]
    VALID_ACTIONS = frozenset()  # type: ignore[assignment]
    get_provenance_tracker = None  # type: ignore[assignment]
    set_provenance_tracker = None  # type: ignore[assignment]
    reset_provenance_tracker = None  # type: ignore[assignment]

# ---- Foundational: metrics ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.metrics import (
        PROMETHEUS_AVAILABLE,
        # Metric objects
        gcv_coordinates_parsed_total,
        gcv_coordinates_validated_total,
        gcv_datum_transforms_total,
        gcv_precision_analyses_total,
        gcv_plausibility_checks_total,
        gcv_reverse_geocodes_total,
        gcv_accuracy_assessments_total,
        gcv_certificates_issued_total,
        gcv_auto_corrections_total,
        gcv_batch_jobs_total,
        gcv_batch_coordinates_processed_total,
        gcv_validation_errors_total,
        gcv_parse_duration_seconds,
        gcv_validation_duration_seconds,
        gcv_batch_duration_seconds,
        gcv_errors_total,
        gcv_active_batch_jobs,
        gcv_avg_accuracy_score,
        # Helper functions
        record_coordinate_parsed,
        record_coordinate_validated,
        record_datum_transform,
        record_precision_analysis,
        record_plausibility_check,
        record_reverse_geocode,
        record_accuracy_assessment,
        record_certificate_issued,
        record_auto_correction,
        record_batch_job,
        record_batch_coordinate_processed,
        record_validation_error,
        observe_parse_duration,
        observe_validation_duration,
        observe_batch_duration,
        record_error,
        set_active_batch_jobs,
        set_avg_accuracy_score,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

# ---- Engine 1: Coordinate Parser ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.coordinate_parser import (
        CoordinateParser,
    )
except ImportError:
    CoordinateParser = None  # type: ignore[assignment,misc]

# ---- Engine 2: Datum Transformer ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.datum_transformer import (
        DatumTransformer,
    )
except ImportError:
    DatumTransformer = None  # type: ignore[assignment,misc]

# ---- Engine 3: Precision Analyzer ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.precision_analyzer import (
        PrecisionAnalyzer,
    )
except ImportError:
    PrecisionAnalyzer = None  # type: ignore[assignment,misc]

# ---- Engine 4: Format Validator ----
try:
    from greenlang.agents.eudr.gps_coordinate_validator.format_validator import (
        FormatValidator,
    )
except ImportError:
    FormatValidator = None  # type: ignore[assignment,misc]


__all__ = [
    # -- Version --
    "__version__",
    "__agent_id__",
    "VERSION",
    # -- Config --
    "GPSCoordinateValidatorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Constants --
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "DEFAULT_QUALITY_WEIGHTS",
    "ACCURACY_TIER_THRESHOLDS",
    "EUDR_MIN_DECIMAL_PLACES",
    "WGS84_SEMI_MAJOR_AXIS",
    "WGS84_INV_FLATTENING",
    "WGS84_FLATTENING",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "CoordinateFormat",
    "GeodeticDatum",
    "PrecisionLevel",
    "ValidationErrorType",
    "PlausibilityCheckResult",
    "AccuracyTier",
    "CorrectionType",
    "ReportFormat",
    "ComplianceStatus",
    "SourceType",
    "LandUseContext",
    "BatchStatus",
    "ElevationSource",
    "HemisphereIndicator",
    # -- Core Models --
    "RawCoordinate",
    "ParsedCoordinate",
    "NormalizedCoordinate",
    "CoordinateValidationError",
    "ValidationResult",
    "PrecisionResult",
    "DatumTransformResult",
    # -- Result Models --
    "PlausibilityResult",
    "ReverseGeocodeResult",
    "AccuracyScore",
    "ComplianceCertificate",
    # -- Request Models --
    "ParseCoordinateRequest",
    "ValidateCoordinateRequest",
    "TransformDatumRequest",
    "AnalyzePrecisionRequest",
    "CheckPlausibilityRequest",
    "ReverseGeocodeRequest",
    "AssessAccuracyRequest",
    "GenerateReportRequest",
    # -- Response Models --
    "ParseCoordinateResponse",
    "ValidateCoordinateResponse",
    "TransformDatumResponse",
    "AnalyzePrecisionResponse",
    "CheckPlausibilityResponse",
    "ReverseGeocodeResponse",
    "AssessAccuracyResponse",
    "BatchValidationResult",
    # -- Provenance --
    "ProvenanceRecord",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Engines --
    "CoordinateParser",
    "DatumTransformer",
    "PrecisionAnalyzer",
    "FormatValidator",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "gcv_coordinates_parsed_total",
    "gcv_coordinates_validated_total",
    "gcv_datum_transforms_total",
    "gcv_precision_analyses_total",
    "gcv_plausibility_checks_total",
    "gcv_reverse_geocodes_total",
    "gcv_accuracy_assessments_total",
    "gcv_certificates_issued_total",
    "gcv_auto_corrections_total",
    "gcv_batch_jobs_total",
    "gcv_batch_coordinates_processed_total",
    "gcv_validation_errors_total",
    "gcv_parse_duration_seconds",
    "gcv_validation_duration_seconds",
    "gcv_batch_duration_seconds",
    "gcv_errors_total",
    "gcv_active_batch_jobs",
    "gcv_avg_accuracy_score",
    "record_coordinate_parsed",
    "record_coordinate_validated",
    "record_datum_transform",
    "record_precision_analysis",
    "record_plausibility_check",
    "record_reverse_geocode",
    "record_accuracy_assessment",
    "record_certificate_issued",
    "record_auto_correction",
    "record_batch_job",
    "record_batch_coordinate_processed",
    "record_validation_error",
    "observe_parse_duration",
    "observe_validation_duration",
    "observe_batch_duration",
    "record_error",
    "set_active_batch_jobs",
    "set_avg_accuracy_score",
]
