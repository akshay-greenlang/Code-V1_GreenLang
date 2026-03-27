# -*- coding: utf-8 -*-
"""
Forest Cover Analysis Agent - AGENT-EUDR-004

Multi-source forest cover analysis engine for EU Deforestation
Regulation (EUDR) Articles 2, 9, 10, and 12 compliance. Analyzes
canopy density from satellite imagery using spectral unmixing and
NDVI regression, classifies forest type against FAO definitions
(>10% canopy, >5m height, >0.5 ha), reconstructs historical forest
cover state at the EUDR cutoff date (December 31, 2020) using
multi-year composites, verifies deforestation-free status by comparing
cutoff and current canopy conditions, estimates canopy height from
GEDI and ICESat-2 LiDAR data, quantifies landscape fragmentation
via patch metrics and effective mesh size, estimates above-ground
biomass and carbon stocks from ESA CCI and allometric models, and
generates provenance-tracked compliance reports for all seven
EUDR-regulated commodities.

This package contains:
    Foundational modules:
        - models: Pydantic v2 data models for canopy density, forest
          type classification, historical reconstruction, deforestation
          verification, height estimation, fragmentation analysis,
          biomass estimation, compliance reporting, and batch analysis
        - config: ForestCoverConfig with GL_EUDR_FCA_ env var support
        - provenance: SHA-256 chain-hashed audit trail tracking
        - metrics: 18 Prometheus self-monitoring metrics (gl_eudr_fca_ prefix)

    Engine modules (to be implemented):
        - canopy_density_mapper: Spectral unmixing and NDVI regression
        - forest_type_classifier: Multi-method forest type classification
        - historical_reconstructor: Cutoff-date baseline reconstruction
        - deforestation_free_verifier: EUDR Article 2 verdict engine
        - canopy_height_modeler: LiDAR and model-based height estimation
        - fragmentation_analyzer: Landscape ecology patch metrics
        - biomass_estimator: AGB and carbon stock estimation
        - compliance_reporter: Regulatory report generation

PRD: PRD-AGENT-EUDR-004
Agent ID: GL-EUDR-FCA-004
Regulation: EU 2023/1115 (EUDR) Article 2(1), Article 9, Article 10, Article 12
Enforcement: December 30, 2025 (large operators); June 30, 2026 (SMEs)

Example:
    >>> from greenlang.agents.eudr.forest_cover_analysis import (
    ...     VerifyDeforestationFreeRequest,
    ...     DeforestationFreeResult,
    ...     DeforestationVerdict,
    ...     ForestType,
    ...     CanopyDensityClass,
    ...     EUDRCommodity,
    ... )
    >>> from datetime import date
    >>> request = VerifyDeforestationFreeRequest(
    ...     plot_id="plot-001",
    ...     polygon_wkt="POLYGON((-62.2 -3.4, -62.1 -3.4, -62.1 -3.5, -62.2 -3.5, -62.2 -3.4))",
    ...     commodity=EUDRCommodity.SOYA,
    ...     include_evidence=True,
    ... )

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

# ---- Foundational: config ----
from greenlang.agents.eudr.forest_cover_analysis.config import (
    ForestCoverConfig,
    get_config,
    set_config,
    reset_config,
)

# ---- Foundational: models ----
from greenlang.agents.eudr.forest_cover_analysis.models import (
    # Constants
    VERSION,
    EUDR_CUTOFF_DATE,
    MAX_BATCH_SIZE,
    FAO_CANOPY_THRESHOLD,
    FAO_HEIGHT_THRESHOLD,
    FAO_AREA_THRESHOLD,
    BIOME_COUNT,
    AGB_CONVERSION_FACTOR,
    # Re-exported from greenlang.agents.data.eudr_traceability.models
    EUDRCommodity,
    # Enumerations
    ForestType,
    CanopyDensityClass,
    DeforestationVerdict,
    DensityMethod,
    ClassificationMethod,
    HeightSource,
    BiomassSource,
    FragmentationLevel,
    ReportFormat,
    AnalysisStatus,
    DataQualityTier,
    # Core models
    CanopyDensityResult,
    ForestClassificationResult,
    HistoricalCoverRecord,
    DeforestationFreeResult,
    CanopyHeightEstimate,
    FragmentationMetrics,
    BiomassEstimate,
    ComplianceReport,
    DataQualityAssessment,
    PlotForestProfile,
    # Request models
    AnalyzeDensityRequest,
    ClassifyForestRequest,
    ReconstructHistoryRequest,
    VerifyDeforestationFreeRequest,
    BatchAnalysisRequest,
    GenerateReportRequest,
    # Response models
    BatchAnalysisResponse,
    BatchProgress,
    AnalysisSummary,
    ForestCoverDashboard,
)

# ---- Foundational: provenance ----
from greenlang.agents.eudr.forest_cover_analysis.provenance import (
    ProvenanceEntry,
    ProvenanceTracker,
    VALID_ENTITY_TYPES,
    VALID_ACTIONS,
    get_provenance_tracker,
    set_provenance_tracker,
    reset_provenance_tracker,
)

# ---- Foundational: metrics ----
from greenlang.agents.eudr.forest_cover_analysis.metrics import (
    PROMETHEUS_AVAILABLE,
    record_density_analysis,
    record_classification,
    record_reconstruction,
    record_verdict,
    record_height_estimate,
    record_fragmentation_analysis,
    record_biomass_estimate,
    record_report_generated,
    record_deforested_plot,
    record_degraded_plot,
    observe_analysis_duration,
    observe_batch_duration,
    set_active_analyses,
    set_avg_canopy_density,
    set_avg_confidence_score,
    record_api_error,
    set_data_quality_score,
    set_forest_area_ha,
)


__all__ = [
    # -- Config --
    "ForestCoverConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -- Version --
    "VERSION",
    # -- Constants --
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "FAO_CANOPY_THRESHOLD",
    "FAO_HEIGHT_THRESHOLD",
    "FAO_AREA_THRESHOLD",
    "BIOME_COUNT",
    "AGB_CONVERSION_FACTOR",
    # -- Re-exported Commodity Enum --
    "EUDRCommodity",
    # -- Enumerations --
    "ForestType",
    "CanopyDensityClass",
    "DeforestationVerdict",
    "DensityMethod",
    "ClassificationMethod",
    "HeightSource",
    "BiomassSource",
    "FragmentationLevel",
    "ReportFormat",
    "AnalysisStatus",
    "DataQualityTier",
    # -- Core Models --
    "CanopyDensityResult",
    "ForestClassificationResult",
    "HistoricalCoverRecord",
    "DeforestationFreeResult",
    "CanopyHeightEstimate",
    "FragmentationMetrics",
    "BiomassEstimate",
    "ComplianceReport",
    "DataQualityAssessment",
    "PlotForestProfile",
    # -- Request Models --
    "AnalyzeDensityRequest",
    "ClassifyForestRequest",
    "ReconstructHistoryRequest",
    "VerifyDeforestationFreeRequest",
    "BatchAnalysisRequest",
    "GenerateReportRequest",
    # -- Response Models --
    "BatchAnalysisResponse",
    "BatchProgress",
    "AnalysisSummary",
    "ForestCoverDashboard",
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
    "record_density_analysis",
    "record_classification",
    "record_reconstruction",
    "record_verdict",
    "record_height_estimate",
    "record_fragmentation_analysis",
    "record_biomass_estimate",
    "record_report_generated",
    "record_deforested_plot",
    "record_degraded_plot",
    "observe_analysis_duration",
    "observe_batch_duration",
    "set_active_analyses",
    "set_avg_canopy_density",
    "set_avg_confidence_score",
    "record_api_error",
    "set_data_quality_score",
    "set_forest_area_ha",
]
