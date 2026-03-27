# -*- coding: utf-8 -*-
"""
Satellite Monitoring Agent - AGENT-EUDR-003

Multi-source satellite imagery analysis engine for EU Deforestation
Regulation (EUDR) Article 9 compliance. Acquires and processes imagery
from Sentinel-2, Landsat 8/9, Sentinel-1 SAR, and Global Forest Watch
alerts to establish spectral baselines at the EUDR cutoff date
(December 31, 2020), detect deforestation and degradation through NDVI
differencing, spectral angle mapping, and time series break detection,
fuse results from multiple sources with weighted confidence scoring,
schedule continuous monitoring with configurable intervals, generate
severity-classified alerts, and assemble provenance-tracked evidence
packages for all seven EUDR-regulated commodities.

This package contains:
    Foundational modules:
        - models: Pydantic v2 data models for satellite scenes, spectral
          indices, baselines, change detection, fusion, monitoring,
          alerts, evidence packages, batch analysis, and summaries
        - config: SatelliteMonitoringConfig with GL_EUDR_SAT_ env var support
        - provenance: SHA-256 chain-hashed audit trail tracking
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_sat_ prefix)

    Engine modules:
        - imagery_acquisition: Scene search, band download, quality assessment
        - spectral_index_calculator: NDVI, EVI, NBR, NDMI, SAVI computation
        - baseline_manager: Dec 31, 2020 baseline establishment and management
        - forest_change_detector: Multi-method deforestation change detection

PRD: PRD-AGENT-EUDR-003
Agent ID: GL-EUDR-SAT-003
Regulation: EU 2023/1115 (EUDR) Article 2(1), Article 9, Article 10
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.satellite_monitoring import (
    ...     SearchScenesRequest,
    ...     ChangeDetectionResult,
    ...     AnalysisLevel,
    ...     SatelliteSource,
    ...     ChangeClassification,
    ...     EUDRCommodity,
    ... )
    >>> from datetime import date
    >>> request = SearchScenesRequest(
    ...     plot_id="plot-001",
    ...     latitude=-3.4653,
    ...     longitude=-62.2159,
    ...     sources=[SatelliteSource.SENTINEL_2],
    ...     date_start=date(2024, 1, 1),
    ...     date_end=date(2024, 12, 31),
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# ---- Foundational: config ----
from greenlang.agents.eudr.satellite_monitoring.config import (
    SatelliteMonitoringConfig,
    get_config,
    set_config,
    reset_config,
)

# ---- Foundational: models ----
from greenlang.agents.eudr.satellite_monitoring.models import (
    # Constants
    VERSION,
    EUDR_CUTOFF_DATE,
    MAX_BATCH_SIZE,
    SENTINEL2_BANDS,
    LANDSAT_BANDS,
    SENTINEL1_SAR_BANDS,
    FOREST_NDVI_THRESHOLDS,
    GFW_ALERT_SOURCES,
    # Re-exported from greenlang.agents.data.eudr_traceability.models
    EUDRCommodity,
    # Enumerations
    SatelliteSource,
    SpectralIndex,
    ForestClassification,
    ChangeClassification,
    DetectionMethod,
    AlertSeverity,
    MonitoringInterval,
    EvidenceFormat,
    CloudFillMethod,
    AnalysisLevel,
    # Core models
    SceneMetadata,
    SceneBand,
    SpectralIndexResult,
    BaselineSnapshot,
    ChangeDetectionResult,
    ChangePixel,
    DataQualityAssessment,
    CloudCoverAnalysis,
    # Result models
    FusionResult,
    MonitoringResult,
    SatelliteAlert,
    EvidencePackage,
    # Request models
    SearchScenesRequest,
    EstablishBaselineRequest,
    DetectChangeRequest,
    CreateMonitoringRequest,
    GenerateEvidenceRequest,
    BatchAnalysisRequest,
    # Response models
    BatchAnalysisResult,
    BatchProgress,
    AlertSummary,
    MonitoringSummary,
)

# ---- Foundational: provenance ----
from greenlang.agents.eudr.satellite_monitoring.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---- Foundational: metrics ----
from greenlang.agents.eudr.satellite_monitoring.metrics import (
    PROMETHEUS_AVAILABLE,
    record_scene_queried,
    record_scene_downloaded,
    record_imagery_download_bytes,
    record_baseline_established,
    record_ndvi_calculation,
    record_change_detection,
    record_deforestation_detected,
    record_alert_generated,
    record_evidence_package,
    record_monitoring_execution,
    record_cloud_gap_fill,
    record_fusion_analysis,
    observe_analysis_duration,
    observe_batch_duration,
    set_active_monitoring_plots,
    set_avg_detection_confidence,
    record_api_error,
    set_data_quality_score,
)

# ---- Engines ----
from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
    ImageryAcquisitionEngine,
    SENTINEL2_BAND_SPECS,
    LANDSAT_BAND_SPECS,
    TILE_GRID,
)

from greenlang.agents.eudr.satellite_monitoring.spectral_index_calculator import (
    SpectralIndexCalculator,
    BIOME_NDVI_THRESHOLDS,
)

from greenlang.agents.eudr.satellite_monitoring.baseline_manager import (
    BaselineManager,
    BIOME_THRESHOLDS,
)

from greenlang.agents.eudr.satellite_monitoring.forest_change_detector import (
    ForestChangeDetector,
    COMMODITY_CHANGE_THRESHOLDS,
)


__all__ = [
    # -- Config --
    "SatelliteMonitoringConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Version --
    "VERSION",
    # -- Constants --
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "SENTINEL2_BANDS",
    "LANDSAT_BANDS",
    "SENTINEL1_SAR_BANDS",
    "FOREST_NDVI_THRESHOLDS",
    "GFW_ALERT_SOURCES",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "SatelliteSource",
    "SpectralIndex",
    "ForestClassification",
    "ChangeClassification",
    "DetectionMethod",
    "AlertSeverity",
    "MonitoringInterval",
    "EvidenceFormat",
    "CloudFillMethod",
    "AnalysisLevel",
    # -- Core Models --
    "SceneMetadata",
    "SceneBand",
    "SpectralIndexResult",
    "BaselineSnapshot",
    "ChangeDetectionResult",
    "ChangePixel",
    "DataQualityAssessment",
    "CloudCoverAnalysis",
    # -- Result Models --
    "FusionResult",
    "MonitoringResult",
    "SatelliteAlert",
    "EvidencePackage",
    # -- Request Models --
    "SearchScenesRequest",
    "EstablishBaselineRequest",
    "DetectChangeRequest",
    "CreateMonitoringRequest",
    "GenerateEvidenceRequest",
    "BatchAnalysisRequest",
    # -- Response Models --
    "BatchAnalysisResult",
    "BatchProgress",
    "AlertSummary",
    "MonitoringSummary",
    # -- Provenance --
    "ProvenanceEntry",
    "ProvenanceTracker",
    "VALID_ENTITY_TYPES",
    "VALID_ACTIONS",
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # -- Metrics --
    "PROMETHEUS_AVAILABLE",
    "record_scene_queried",
    "record_scene_downloaded",
    "record_imagery_download_bytes",
    "record_baseline_established",
    "record_ndvi_calculation",
    "record_change_detection",
    "record_deforestation_detected",
    "record_alert_generated",
    "record_evidence_package",
    "record_monitoring_execution",
    "record_cloud_gap_fill",
    "record_fusion_analysis",
    "observe_analysis_duration",
    "observe_batch_duration",
    "set_active_monitoring_plots",
    "set_avg_detection_confidence",
    "record_api_error",
    "set_data_quality_score",
    # -- Engines --
    "ImageryAcquisitionEngine",
    "SpectralIndexCalculator",
    "BaselineManager",
    "ForestChangeDetector",
    # -- Engine reference data --
    "SENTINEL2_BAND_SPECS",
    "LANDSAT_BAND_SPECS",
    "TILE_GRID",
    "BIOME_NDVI_THRESHOLDS",
    "BIOME_THRESHOLDS",
    "COMMODITY_CHANGE_THRESHOLDS",
]
