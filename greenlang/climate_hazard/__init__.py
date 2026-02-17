# -*- coding: utf-8 -*-
"""
GL-DATA-GEO-002: GreenLang Climate Hazard Connector Agent Service SDK
======================================================================

This package provides climate hazard data ingestion, risk index
calculation, scenario projection, exposure assessment, vulnerability
scoring, compliance reporting, and end-to-end pipeline orchestration
for the GreenLang framework. It supports:

- Multi-source hazard data ingestion (NOAA, Copernicus, World Bank,
  NASA, IPCC, national agencies, satellite, ground stations, model
  outputs) with hazard type classification (flood, drought, wildfire,
  heat wave, cold wave, storm, sea level rise, tropical cyclone,
  landslide, water stress, precipitation change, temperature change,
  compound)
- Composite risk index calculation with configurable weights
  (probability, intensity, frequency, duration) and risk
  classification (extreme, high, medium, low, negligible)
- Multi-hazard and compound risk assessment with location comparison
- IPCC scenario projection under SSP pathways (SSP1-1.9, SSP1-2.6,
  SSP2-4.5, SSP3-7.0, SSP5-8.5) and legacy RCP pathways (RCP2.6,
  RCP4.5, RCP6.0, RCP8.5) with configurable time horizons (2030,
  2050, 2100)
- Asset-level and portfolio-level exposure assessment with hotspot
  identification and supply chain exposure mapping
- Vulnerability scoring with exposure, sensitivity, and adaptive
  capacity weighting, sector-level analysis, and residual risk
  calculation
- TCFD, CSRD/ESRS, and EU Taxonomy compliance report generation
  with evidence summaries and provenance chain hashing (SHA-256)
- 7-stage processing pipeline (ingestion, risk calculation, scenario
  projection, exposure assessment, vulnerability scoring, reporting,
  full pipeline)
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics with gl_chc_ prefix for observability
- FastAPI REST API with 20 endpoints at /api/v1/climate-hazard
- Thread-safe configuration with GL_CLIMATE_HAZARD_ env prefix

Key Components:
    - config: ClimateHazardConfig with GL_CLIMATE_HAZARD_ env prefix
    - models: Pydantic v2 models (12 enums, 14+ data models)
    - hazard_database: Hazard data source registry and ingestion engine
    - risk_index: Composite risk index calculation engine
    - scenario_projector: Climate scenario projection engine (SSP/RCP)
    - exposure_assessor: Asset and portfolio exposure assessment engine
    - vulnerability_scorer: Vulnerability and adaptive capacity engine
    - compliance_reporter: TCFD/CSRD/EU Taxonomy report generation engine
    - hazard_pipeline: End-to-end pipeline orchestration engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics with gl_chc_ prefix
    - api: FastAPI HTTP service with 20 endpoints
    - setup: ClimateHazardService facade

Example:
    >>> from greenlang.climate_hazard import ClimateHazardService
    >>> service = ClimateHazardService()
    >>> result = service.calculate_risk_index(
    ...     location_id="loc_001",
    ...     hazard_type="flood",
    ...     scenario="SSP2-4.5",
    ... )
    >>> print(result["risk_classification"], result["composite_score"])

Agent ID: GL-DATA-GEO-002
Agent Name: Climate Hazard Connector Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-GEO-002"
__agent_name__ = "Climate Hazard Connector Agent"

# SDK availability flag
CLIMATE_HAZARD_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.climate_hazard.config import (
    ClimateHazardConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.climate_hazard.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.climate_hazard.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    chc_hazard_data_ingested_total,
    chc_risk_indices_calculated_total,
    chc_scenario_projections_total,
    chc_exposure_assessments_total,
    chc_vulnerability_scores_total,
    chc_reports_generated_total,
    chc_pipeline_runs_total,
    chc_active_sources,
    chc_active_assets,
    chc_high_risk_locations,
    chc_ingestion_duration_seconds,
    chc_pipeline_duration_seconds,
    # Helper functions
    record_ingestion,
    record_risk_calculation,
    record_projection,
    record_exposure,
    record_vulnerability,
    record_report,
    record_pipeline,
    set_active_sources,
    set_active_assets,
    set_high_risk,
    observe_ingestion_duration,
    observe_pipeline_duration,
)

# ---------------------------------------------------------------------------
# Models (enums, data models) - optional import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.climate_hazard.models import (
        # Enumerations
        HazardType,
        HazardSeverity,
        ClimateScenario,
        TimeHorizon,
        RiskClassification,
        AssetType,
        ExposureLevel,
        VulnerabilityLevel,
        AdaptiveCapacity,
        ReportType,
        ReportFormat,
        PipelineStage,
        # Core data models
        HazardSource,
        HazardDataRecord,
        HazardEvent,
        RiskIndex,
        MultiHazardIndex,
        LocationComparison,
        ScenarioProjection,
        Asset,
        ExposureAssessment,
        PortfolioExposure,
        VulnerabilityScore,
        ComplianceReport,
        PipelineResult,
        PipelineRunConfig,
    )
except ImportError:
    HazardType = None  # type: ignore[assignment, misc]
    HazardSeverity = None  # type: ignore[assignment, misc]
    ClimateScenario = None  # type: ignore[assignment, misc]
    TimeHorizon = None  # type: ignore[assignment, misc]
    RiskClassification = None  # type: ignore[assignment, misc]
    AssetType = None  # type: ignore[assignment, misc]
    ExposureLevel = None  # type: ignore[assignment, misc]
    VulnerabilityLevel = None  # type: ignore[assignment, misc]
    AdaptiveCapacity = None  # type: ignore[assignment, misc]
    ReportType = None  # type: ignore[assignment, misc]
    ReportFormat = None  # type: ignore[assignment, misc]
    PipelineStage = None  # type: ignore[assignment, misc]
    HazardSource = None  # type: ignore[assignment, misc]
    HazardDataRecord = None  # type: ignore[assignment, misc]
    HazardEvent = None  # type: ignore[assignment, misc]
    RiskIndex = None  # type: ignore[assignment, misc]
    MultiHazardIndex = None  # type: ignore[assignment, misc]
    LocationComparison = None  # type: ignore[assignment, misc]
    ScenarioProjection = None  # type: ignore[assignment, misc]
    Asset = None  # type: ignore[assignment, misc]
    ExposureAssessment = None  # type: ignore[assignment, misc]
    PortfolioExposure = None  # type: ignore[assignment, misc]
    VulnerabilityScore = None  # type: ignore[assignment, misc]
    ComplianceReport = None  # type: ignore[assignment, misc]
    PipelineResult = None  # type: ignore[assignment, misc]
    PipelineRunConfig = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    from greenlang.climate_hazard.hazard_database import HazardDatabaseEngine
except ImportError:
    HazardDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.risk_index import RiskIndexEngine
except ImportError:
    RiskIndexEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.scenario_projector import ScenarioProjectorEngine
except ImportError:
    ScenarioProjectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.exposure_assessor import ExposureAssessorEngine
except ImportError:
    ExposureAssessorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.vulnerability_scorer import VulnerabilityScorerEngine
except ImportError:
    VulnerabilityScorerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.compliance_reporter import ComplianceReporterEngine
except ImportError:
    ComplianceReporterEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.hazard_pipeline import HazardPipelineEngine
except ImportError:
    HazardPipelineEngine = None  # type: ignore[assignment, misc]

# Pipeline stages constant
try:
    from greenlang.climate_hazard.hazard_pipeline import PIPELINE_STAGES
except ImportError:
    PIPELINE_STAGES = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade and response models
# ---------------------------------------------------------------------------
try:
    from greenlang.climate_hazard.setup import (
        ClimateHazardService,
        configure_climate_hazard,
        get_service,
        get_router,
        # Response models
        SourceResponse,
        HazardDataResponse,
        HazardEventResponse,
        RiskIndexResponse,
        MultiHazardResponse,
        LocationComparisonResponse,
        ScenarioResponse,
        AssetResponse,
        ExposureResponse,
        PortfolioExposureResponse,
        VulnerabilityResponse,
        ReportResponse,
        PipelineResponse,
        HealthResponse,
    )
except ImportError:
    ClimateHazardService = None  # type: ignore[assignment, misc]
    configure_climate_hazard = None  # type: ignore[assignment, misc]
    get_service = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]
    SourceResponse = None  # type: ignore[assignment, misc]
    HazardDataResponse = None  # type: ignore[assignment, misc]
    HazardEventResponse = None  # type: ignore[assignment, misc]
    RiskIndexResponse = None  # type: ignore[assignment, misc]
    MultiHazardResponse = None  # type: ignore[assignment, misc]
    LocationComparisonResponse = None  # type: ignore[assignment, misc]
    ScenarioResponse = None  # type: ignore[assignment, misc]
    AssetResponse = None  # type: ignore[assignment, misc]
    ExposureResponse = None  # type: ignore[assignment, misc]
    PortfolioExposureResponse = None  # type: ignore[assignment, misc]
    VulnerabilityResponse = None  # type: ignore[assignment, misc]
    ReportResponse = None  # type: ignore[assignment, misc]
    PipelineResponse = None  # type: ignore[assignment, misc]
    HealthResponse = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from gis_connector (GL-DATA-GEO-001)
# ---------------------------------------------------------------------------
try:
    from greenlang.gis_connector.spatial_analyzer import SpatialAnalyzerEngine
except ImportError:
    SpatialAnalyzerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.gis_connector.land_cover import LandCoverEngine
except ImportError:
    LandCoverEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.gis_connector.models import (
        CoordinateSystem,
        GeometryType,
    )
except ImportError:
    CoordinateSystem = None  # type: ignore[assignment, misc]
    GeometryType = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "CLIMATE_HAZARD_SDK_AVAILABLE",
    # Configuration
    "ClimateHazardConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "chc_hazard_data_ingested_total",
    "chc_risk_indices_calculated_total",
    "chc_scenario_projections_total",
    "chc_exposure_assessments_total",
    "chc_vulnerability_scores_total",
    "chc_reports_generated_total",
    "chc_pipeline_runs_total",
    "chc_active_sources",
    "chc_active_assets",
    "chc_high_risk_locations",
    "chc_ingestion_duration_seconds",
    "chc_pipeline_duration_seconds",
    # Metric helper functions
    "record_ingestion",
    "record_risk_calculation",
    "record_projection",
    "record_exposure",
    "record_vulnerability",
    "record_report",
    "record_pipeline",
    "set_active_sources",
    "set_active_assets",
    "set_high_risk",
    "observe_ingestion_duration",
    "observe_pipeline_duration",
    # Enumerations
    "HazardType",
    "HazardSeverity",
    "ClimateScenario",
    "TimeHorizon",
    "RiskClassification",
    "AssetType",
    "ExposureLevel",
    "VulnerabilityLevel",
    "AdaptiveCapacity",
    "ReportType",
    "ReportFormat",
    "PipelineStage",
    # Core data models
    "HazardSource",
    "HazardDataRecord",
    "HazardEvent",
    "RiskIndex",
    "MultiHazardIndex",
    "LocationComparison",
    "ScenarioProjection",
    "Asset",
    "ExposureAssessment",
    "PortfolioExposure",
    "VulnerabilityScore",
    "ComplianceReport",
    "PipelineResult",
    "PipelineRunConfig",
    # Core engines (Layer 2)
    "HazardDatabaseEngine",
    "RiskIndexEngine",
    "ScenarioProjectorEngine",
    "ExposureAssessorEngine",
    "VulnerabilityScorerEngine",
    "ComplianceReporterEngine",
    "HazardPipelineEngine",
    # Pipeline stages
    "PIPELINE_STAGES",
    # Service setup facade
    "ClimateHazardService",
    "configure_climate_hazard",
    "get_service",
    "get_router",
    # Response models
    "SourceResponse",
    "HazardDataResponse",
    "HazardEventResponse",
    "RiskIndexResponse",
    "MultiHazardResponse",
    "LocationComparisonResponse",
    "ScenarioResponse",
    "AssetResponse",
    "ExposureResponse",
    "PortfolioExposureResponse",
    "VulnerabilityResponse",
    "ReportResponse",
    "PipelineResponse",
    "HealthResponse",
    # Layer 1 re-exports (gis_connector GL-DATA-GEO-001)
    "SpatialAnalyzerEngine",
    "LandCoverEngine",
    "CoordinateSystem",
    "GeometryType",
]
