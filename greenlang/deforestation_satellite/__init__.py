# -*- coding: utf-8 -*-
"""
GL-DATA-GEO-003: GreenLang Deforestation Satellite Connector Agent Service SDK
================================================================================

This package provides satellite-based deforestation monitoring and EUDR
compliance assessment capabilities for the GreenLang platform. It supports:

- Multi-satellite imagery acquisition (Sentinel-2, Landsat-8/9, MODIS,
  Harmonized Sentinel/Landsat)
- 7 vegetation indices (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI)
- Temporal change detection with configurable NDVI/NBR thresholds
  (clear-cut, degradation, partial loss, regrowth)
- 10 land cover classes with forest/non-forest classification
- Country-specific forest definitions (FAO, national laws)
- EUDR baseline assessment with cutoff date enforcement (2020-12-31)
- 8 forest status categories for regulatory compliance
- 5 deforestation risk levels (low through violation)
- External alert integration (GLAD, RADD, FIRMS, GFW) with
  spatial deduplication and confidence filtering
- Compliance report generation with evidence summaries and
  provenance chain hashing (SHA-256)
- Recurring monitoring jobs (on-demand, weekly, monthly, quarterly)
- 7-stage processing pipeline (init, acquisition, index calculation,
  classification, change detection, alert integration, report generation)
- Trend analysis over multi-year periods
- Country-level risk profiling with commodity-specific adjustments
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_DEFORESTATION_SAT_ env prefix

Key Components:
    - config: DeforestationSatelliteConfig with GL_DEFORESTATION_SAT_ env prefix
    - models: Pydantic v2 models (12 enums, 14 data models, 6 request models)
    - satellite_engine: Multi-source satellite imagery acquisition engine
    - index_engine: Vegetation index computation engine (7 indices)
    - change_engine: Temporal change detection and classification engine
    - classification_engine: Forest/land cover classification engine
    - alert_engine: Deforestation alert integration and deduplication engine
    - baseline_engine: EUDR baseline assessment and compliance engine
    - monitoring_engine: Recurring monitoring job management engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics for observability
    - setup: DeforestationSatelliteService facade

Example:
    >>> from greenlang.deforestation_satellite import DeforestationSatelliteService
    >>> service = DeforestationSatelliteService()
    >>> result = service.check_baseline(latitude=-3.4, longitude=-60.0, country_iso3="BRA")
    >>> print(result.risk_level, result.is_eudr_compliant)

Agent ID: GL-DATA-GEO-003
Agent Name: Deforestation Satellite Connector Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-GEO-003"
__agent_name__ = "Deforestation Satellite Connector Agent"

# SDK availability flag
DEFORESTATION_SATELLITE_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.deforestation_satellite.config import (
    DeforestationSatelliteConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, data models, request models)
# ---------------------------------------------------------------------------
from greenlang.deforestation_satellite.models import (
    # Enumerations
    SatelliteSource,
    VegetationIndex,
    ChangeType,
    LandCoverClass,
    ForestStatus,
    DeforestationRisk,
    ComplianceStatus,
    AlertSource,
    AlertConfidence,
    AlertSeverity,
    PipelineStage,
    MonitoringFrequency,
    # Core data models
    SatelliteScene,
    VegetationIndexResult,
    ChangeDetectionResult,
    ForestClassification,
    DeforestationAlert,
    ForestDefinition,
    BaselineAssessment,
    ComplianceReport,
    MonitoringJob,
    PipelineResult,
    AlertAggregation,
    TrendAnalysis,
    DeforestationStatistics,
    CountryRiskProfile,
    # Request models
    AcquireSatelliteRequest,
    DetectChangeRequest,
    CheckBaselineRequest,
    CheckBaselinePolygonRequest,
    QueryAlertsRequest,
    StartMonitoringRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.deforestation_satellite.engines import (
    SatelliteEngine,
    IndexEngine,
    ChangeEngine,
    ClassificationEngine,
    AlertEngine,
    BaselineEngine,
    MonitoringEngine,
)
from greenlang.deforestation_satellite.engines import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.deforestation_satellite.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    deforestation_scenes_acquired_total,
    deforestation_scene_acquisition_duration_seconds,
    deforestation_indices_computed_total,
    deforestation_changes_detected_total,
    deforestation_classifications_total,
    deforestation_alerts_processed_total,
    deforestation_baseline_assessments_total,
    deforestation_compliance_reports_total,
    deforestation_monitoring_jobs_active,
    deforestation_pipeline_stage_duration_seconds,
    deforestation_processing_errors_total,
    deforestation_cache_hit_rate,
    # Helper functions
    record_scene_acquisition,
    record_index_computation,
    record_change_detection,
    record_classification,
    record_alert_processed,
    record_baseline_assessment,
    record_compliance_report,
    update_active_monitoring_jobs,
    record_pipeline_stage,
    record_processing_error,
    update_cache_hit_rate,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.deforestation_satellite.setup import (
    DeforestationSatelliteService,
    configure_deforestation_satellite,
    get_deforestation_satellite,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "DEFORESTATION_SATELLITE_SDK_AVAILABLE",
    # Configuration
    "DeforestationSatelliteConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "SatelliteSource",
    "VegetationIndex",
    "ChangeType",
    "LandCoverClass",
    "ForestStatus",
    "DeforestationRisk",
    "ComplianceStatus",
    "AlertSource",
    "AlertConfidence",
    "AlertSeverity",
    "PipelineStage",
    "MonitoringFrequency",
    # Core data models
    "SatelliteScene",
    "VegetationIndexResult",
    "ChangeDetectionResult",
    "ForestClassification",
    "DeforestationAlert",
    "ForestDefinition",
    "BaselineAssessment",
    "ComplianceReport",
    "MonitoringJob",
    "PipelineResult",
    "AlertAggregation",
    "TrendAnalysis",
    "DeforestationStatistics",
    "CountryRiskProfile",
    # Request models
    "AcquireSatelliteRequest",
    "DetectChangeRequest",
    "CheckBaselineRequest",
    "CheckBaselinePolygonRequest",
    "QueryAlertsRequest",
    "StartMonitoringRequest",
    # Core engines
    "SatelliteEngine",
    "IndexEngine",
    "ChangeEngine",
    "ClassificationEngine",
    "AlertEngine",
    "BaselineEngine",
    "MonitoringEngine",
    "ProvenanceTracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "deforestation_scenes_acquired_total",
    "deforestation_scene_acquisition_duration_seconds",
    "deforestation_indices_computed_total",
    "deforestation_changes_detected_total",
    "deforestation_classifications_total",
    "deforestation_alerts_processed_total",
    "deforestation_baseline_assessments_total",
    "deforestation_compliance_reports_total",
    "deforestation_monitoring_jobs_active",
    "deforestation_pipeline_stage_duration_seconds",
    "deforestation_processing_errors_total",
    "deforestation_cache_hit_rate",
    # Metric helper functions
    "record_scene_acquisition",
    "record_index_computation",
    "record_change_detection",
    "record_classification",
    "record_alert_processed",
    "record_baseline_assessment",
    "record_compliance_report",
    "update_active_monitoring_jobs",
    "record_pipeline_stage",
    "record_processing_error",
    "update_cache_hit_rate",
    # Service setup facade
    "DeforestationSatelliteService",
    "configure_deforestation_satellite",
    "get_deforestation_satellite",
    "get_router",
]
