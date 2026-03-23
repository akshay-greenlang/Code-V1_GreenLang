# -*- coding: utf-8 -*-
"""
Deforestation Satellite Connector Agent Data Models - AGENT-DATA-007: GL-DATA-GEO-003

Pydantic v2 data models for the Deforestation Satellite Connector Agent SDK.
Defines all enumerations, core data models, and request wrappers required for
satellite-based deforestation monitoring operations including:

- Satellite scene acquisition and metadata
- Vegetation index computation (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI)
- Change detection with temporal comparison
- Forest classification and land cover typing
- Deforestation alert integration (GLAD, RADD, FIRMS, GFW)
- EUDR baseline assessment and compliance reporting
- Monitoring job lifecycle management
- Pipeline stage tracking and result aggregation
- Trend analysis and deforestation statistics
- Country-level risk profiling

Models:
    - Enumerations (12): SatelliteSource, VegetationIndex, ChangeType,
        LandCoverClass, ForestStatus, DeforestationRisk, ComplianceStatus,
        AlertSource, AlertConfidence, AlertSeverity, PipelineStage,
        MonitoringFrequency
    - Core data models (14): SatelliteScene, VegetationIndexResult,
        ChangeDetectionResult, ForestClassification, DeforestationAlert,
        ForestDefinition, BaselineAssessment, ComplianceReport,
        MonitoringJob, PipelineResult, AlertAggregation, TrendAnalysis,
        DeforestationStatistics, CountryRiskProfile
    - Request models (6): AcquireSatelliteRequest, DetectChangeRequest,
        CheckBaselineRequest, CheckBaselinePolygonRequest,
        QueryAlertsRequest, StartMonitoringRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_id(prefix: str) -> str:
    """Generate a short prefixed unique identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# =============================================================================
# Enumerations
# =============================================================================


class SatelliteSource(str, Enum):
    """Satellite data sources supported for image acquisition.

    Covers the primary open-access Earth observation satellites used
    for deforestation monitoring at various spatial and temporal
    resolutions.
    """

    SENTINEL2 = "sentinel2"
    LANDSAT8 = "landsat8"
    LANDSAT9 = "landsat9"
    MODIS = "modis"
    HARMONIZED = "harmonized"


class VegetationIndex(str, Enum):
    """Spectral vegetation indices computable from satellite imagery.

    Each index measures a different aspect of vegetation health,
    moisture content, or burn severity.
    """

    NDVI = "ndvi"
    EVI = "evi"
    NDWI = "ndwi"
    NBR = "nbr"
    SAVI = "savi"
    MSAVI = "msavi"
    NDMI = "ndmi"


class ChangeType(str, Enum):
    """Types of land cover change detected between temporal windows.

    Categorises the nature and severity of detected vegetation
    change from satellite imagery comparison.
    """

    NO_CHANGE = "no_change"
    CLEAR_CUT = "clear_cut"
    DEGRADATION = "degradation"
    PARTIAL_LOSS = "partial_loss"
    REGROWTH = "regrowth"


class LandCoverClass(str, Enum):
    """Land cover classification types for pixel-level labelling.

    Based on commonly used land cover taxonomies including FAO
    and Copernicus Global Land Cover schemes.
    """

    DENSE_FOREST = "dense_forest"
    OPEN_FOREST = "open_forest"
    SHRUBLAND = "shrubland"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    BARE_SOIL = "bare_soil"
    WATER = "water"
    URBAN = "urban"
    WETLAND = "wetland"
    UNKNOWN = "unknown"


class ForestStatus(str, Enum):
    """Forest status classification for EUDR compliance assessment.

    Determines the regulatory status of a land parcel relative to
    the EUDR cutoff date and current forest cover condition.
    """

    FOREST = "forest"
    NON_FOREST = "non_forest"
    DEFORESTED_PRE_CUTOFF = "deforested_pre_cutoff"
    DEFORESTED_POST_CUTOFF = "deforested_post_cutoff"
    DEGRADED = "degraded"
    REGENERATING = "regenerating"
    PLANTATION = "plantation"
    UNKNOWN = "unknown"


class DeforestationRisk(str, Enum):
    """Risk level classification for deforestation compliance.

    Ordered severity levels from low (compliant) through critical
    to violation (confirmed non-compliance).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    VIOLATION = "violation"


class ComplianceStatus(str, Enum):
    """EUDR compliance determination status.

    Final compliance verdict for a geographic area or supply chain
    node based on deforestation risk assessment.
    """

    COMPLIANT = "compliant"
    REVIEW_REQUIRED = "review_required"
    NON_COMPLIANT = "non_compliant"


class AlertSource(str, Enum):
    """External and internal deforestation alert data sources.

    Integrates alerts from major global forest monitoring systems
    and internal change detection pipelines.
    """

    GLAD = "glad"
    RADD = "radd"
    FIRMS = "firms"
    GFW = "gfw"
    INTERNAL = "internal"


class AlertConfidence(str, Enum):
    """Confidence levels for deforestation alerts.

    Mirrors the confidence classification used by GLAD and GFW
    alert systems.
    """

    LOW = "low"
    NOMINAL = "nominal"
    HIGH = "high"


class AlertSeverity(str, Enum):
    """Severity levels for deforestation alerts.

    Classifies the urgency of detected deforestation events
    for prioritised response and escalation.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PipelineStage(str, Enum):
    """Stages of the deforestation monitoring pipeline.

    Defines the sequential processing stages from satellite image
    acquisition through to compliance report generation.
    """

    INITIALIZATION = "initialization"
    IMAGE_ACQUISITION = "image_acquisition"
    INDEX_CALCULATION = "index_calculation"
    CLASSIFICATION = "classification"
    CHANGE_DETECTION = "change_detection"
    ALERT_INTEGRATION = "alert_integration"
    REPORT_GENERATION = "report_generation"


class MonitoringFrequency(str, Enum):
    """Supported monitoring schedule frequencies.

    Defines how often a monitoring job re-executes the
    deforestation detection pipeline for a tracked area.
    """

    ON_DEMAND = "on_demand"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


# =============================================================================
# Core Data Models
# =============================================================================


class SatelliteScene(BaseModel):
    """Metadata for an acquired satellite scene.

    Represents a single satellite image tile with acquisition date,
    cloud cover, spectral bands, spatial extent, and resolution.

    Attributes:
        scene_id: Unique identifier for this satellite scene.
        satellite: Satellite source that captured the image.
        acquisition_date: Date of image acquisition (ISO YYYY-MM-DD).
        cloud_cover_percent: Cloud cover percentage of the scene (0-100).
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat].
        bands: Dictionary of available spectral bands and their metadata.
        resolution_m: Spatial resolution in meters per pixel.
        crs: Coordinate reference system of the scene (e.g. EPSG:32633).
        tile_id: Satellite-specific tile identifier (e.g. Sentinel-2 MGRS tile).
        metadata: Additional scene metadata from the data provider.
    """

    scene_id: str = Field(
        default="",
        description="Unique identifier for this satellite scene",
    )
    satellite: str = Field(
        default="sentinel2",
        description="Satellite source that captured the image",
    )
    acquisition_date: str = Field(
        default="",
        description="Date of image acquisition (ISO YYYY-MM-DD)",
    )
    cloud_cover_percent: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cloud cover percentage of the scene (0-100)",
    )
    bbox: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0],
        description="Bounding box as [min_lon, min_lat, max_lon, max_lat]",
    )
    bands: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of available spectral bands and their metadata",
    )
    resolution_m: float = Field(
        default=10.0, ge=0.0,
        description="Spatial resolution in meters per pixel",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate reference system of the scene",
    )
    tile_id: str = Field(
        default="",
        description="Satellite-specific tile identifier",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional scene metadata from the data provider",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.scene_id:
            self.scene_id = _new_id("SCN")


class VegetationIndexResult(BaseModel):
    """Result of a vegetation index computation over a satellite scene.

    Contains the computed index type, pixel value array, and summary
    statistics for the area of interest.

    Attributes:
        index_type: Type of vegetation index computed.
        values: List of pixel-level index values.
        min_value: Minimum index value across all pixels.
        max_value: Maximum index value across all pixels.
        mean_value: Mean index value across all pixels.
        std_value: Standard deviation of index values.
        computation_date: Date of computation (ISO YYYY-MM-DD).
    """

    index_type: str = Field(
        default="ndvi",
        description="Type of vegetation index computed",
    )
    values: List[float] = Field(
        default_factory=list,
        description="List of pixel-level index values",
    )
    min_value: float = Field(
        default=0.0,
        description="Minimum index value across all pixels",
    )
    max_value: float = Field(
        default=0.0,
        description="Maximum index value across all pixels",
    )
    mean_value: float = Field(
        default=0.0,
        description="Mean index value across all pixels",
    )
    std_value: float = Field(
        default=0.0,
        description="Standard deviation of index values",
    )
    computation_date: str = Field(
        default="",
        description="Date of computation (ISO YYYY-MM-DD)",
    )

    model_config = ConfigDict(from_attributes=True)


class ChangeDetectionResult(BaseModel):
    """Result of a temporal change detection analysis.

    Compares vegetation indices between two temporal windows to
    identify and classify land cover changes.

    Attributes:
        change_id: Unique identifier for this change detection result.
        change_type: Classified type of change detected.
        pre_date: Start date of the pre-change temporal window (ISO YYYY-MM-DD).
        post_date: Start date of the post-change temporal window (ISO YYYY-MM-DD).
        pre_ndvi: Mean NDVI value in the pre-change window.
        post_ndvi: Mean NDVI value in the post-change window.
        delta_ndvi: Change in NDVI between pre and post windows.
        delta_nbr: Optional change in NBR between pre and post windows.
        area_ha: Affected area in hectares.
        confidence: Detection confidence score (0.0-1.0).
        pixel_count: Number of pixels contributing to the detection.
    """

    change_id: str = Field(
        default="",
        description="Unique identifier for this change detection result",
    )
    change_type: str = Field(
        default="no_change",
        description="Classified type of change detected",
    )
    pre_date: str = Field(
        default="",
        description="Start date of the pre-change temporal window (ISO YYYY-MM-DD)",
    )
    post_date: str = Field(
        default="",
        description="Start date of the post-change temporal window (ISO YYYY-MM-DD)",
    )
    pre_ndvi: float = Field(
        default=0.0,
        description="Mean NDVI value in the pre-change window",
    )
    post_ndvi: float = Field(
        default=0.0,
        description="Mean NDVI value in the post-change window",
    )
    delta_ndvi: float = Field(
        default=0.0,
        description="Change in NDVI between pre and post windows",
    )
    delta_nbr: Optional[float] = Field(
        None,
        description="Optional change in NBR between pre and post windows",
    )
    area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Affected area in hectares",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Detection confidence score (0.0-1.0)",
    )
    pixel_count: int = Field(
        default=0, ge=0,
        description="Number of pixels contributing to the detection",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.change_id:
            self.change_id = _new_id("CHG")


class ForestClassification(BaseModel):
    """Result of a forest/land cover classification for a geographic area.

    Contains the classified land cover type, tree cover percentage,
    canopy height, and classification confidence.

    Attributes:
        classification_id: Unique identifier for this classification.
        land_cover_class: Classified land cover type.
        tree_cover_percent: Percentage of tree cover in the area (0-100).
        is_forest: Whether the area qualifies as forest under applicable
            national or international definitions.
        canopy_height_m: Optional estimated canopy height in meters.
        confidence: Classification confidence score (0.0-1.0).
        method: Classification method used (e.g. random_forest, threshold).
        pixel_count: Number of pixels classified.
    """

    classification_id: str = Field(
        default="",
        description="Unique identifier for this classification",
    )
    land_cover_class: str = Field(
        default="unknown",
        description="Classified land cover type",
    )
    tree_cover_percent: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of tree cover in the area (0-100)",
    )
    is_forest: bool = Field(
        default=False,
        description="Whether the area qualifies as forest",
    )
    canopy_height_m: Optional[float] = Field(
        None, ge=0.0,
        description="Optional estimated canopy height in meters",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Classification confidence score (0.0-1.0)",
    )
    method: str = Field(
        default="threshold",
        description="Classification method used",
    )
    pixel_count: int = Field(
        default=0, ge=0,
        description="Number of pixels classified",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.classification_id:
            self.classification_id = _new_id("FCL")


class DeforestationAlert(BaseModel):
    """A deforestation alert from external or internal monitoring systems.

    Represents a single alert detection with geographic coordinates,
    affected area, confidence, severity, and EUDR cutoff status.

    Attributes:
        alert_id: Unique identifier for this alert.
        source: Alert data source (GLAD, RADD, FIRMS, GFW, internal).
        detection_date: Date the alert was detected (ISO YYYY-MM-DD).
        latitude: Latitude of the alert centroid.
        longitude: Longitude of the alert centroid.
        area_ha: Affected area in hectares.
        confidence: Alert confidence level (low, nominal, high).
        severity: Alert severity level (low, medium, high, critical).
        alert_type: Type of deforestation activity detected.
        is_post_cutoff: Whether the detected event occurred after the
            EUDR cutoff date (2020-12-31).
        metadata: Additional alert metadata from the source system.
    """

    alert_id: str = Field(
        default="",
        description="Unique identifier for this alert",
    )
    source: str = Field(
        default="internal",
        description="Alert data source",
    )
    detection_date: str = Field(
        default="",
        description="Date the alert was detected (ISO YYYY-MM-DD)",
    )
    latitude: float = Field(
        default=0.0,
        description="Latitude of the alert centroid",
    )
    longitude: float = Field(
        default=0.0,
        description="Longitude of the alert centroid",
    )
    area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Affected area in hectares",
    )
    confidence: str = Field(
        default="nominal",
        description="Alert confidence level",
    )
    severity: str = Field(
        default="medium",
        description="Alert severity level",
    )
    alert_type: str = Field(
        default="",
        description="Type of deforestation activity detected",
    )
    is_post_cutoff: bool = Field(
        default=False,
        description="Whether the event occurred after the EUDR cutoff date",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert metadata from the source system",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.alert_id:
            self.alert_id = _new_id("ALT")


class ForestDefinition(BaseModel):
    """Country-specific or international forest definition parameters.

    Captures the minimum thresholds that define what constitutes
    a 'forest' for a given jurisdiction, used for consistent
    classification across different regulatory regimes.

    Attributes:
        country_iso3: ISO 3166-1 alpha-3 country code.
        region: Optional sub-national region name.
        min_tree_cover_percent: Minimum tree canopy cover percentage (0-100).
        min_tree_height_meters: Minimum tree height in meters.
        min_area_hectares: Minimum contiguous area in hectares.
        includes_plantations: Whether timber plantations count as forest.
        includes_agroforestry: Whether agroforestry systems count as forest.
        source: Source of the forest definition (e.g. FAO, national law).
        definition_year: Year the definition was established or updated.
    """

    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    region: Optional[str] = Field(
        None,
        description="Optional sub-national region name",
    )
    min_tree_cover_percent: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Minimum tree canopy cover percentage (0-100)",
    )
    min_tree_height_meters: float = Field(
        default=5.0, ge=0.0,
        description="Minimum tree height in meters",
    )
    min_area_hectares: float = Field(
        default=0.5, ge=0.0,
        description="Minimum contiguous area in hectares",
    )
    includes_plantations: bool = Field(
        default=False,
        description="Whether timber plantations count as forest",
    )
    includes_agroforestry: bool = Field(
        default=False,
        description="Whether agroforestry systems count as forest",
    )
    source: str = Field(
        default="FAO",
        description="Source of the forest definition",
    )
    definition_year: int = Field(
        default=2020,
        description="Year the definition was established or updated",
    )

    model_config = ConfigDict(from_attributes=True)


class BaselineAssessment(BaseModel):
    """EUDR baseline assessment result for a geographic coordinate or area.

    Provides a comprehensive deforestation risk assessment including
    forest status, EUDR compliance determination, risk scoring,
    forest cover change analysis, and supporting evidence.

    Attributes:
        assessment_id: Unique identifier for this baseline assessment.
        coordinate_lat: Latitude of the assessed point.
        coordinate_lon: Longitude of the assessed point.
        country_iso3: ISO 3166-1 alpha-3 country code.
        forest_status: Current forest status classification.
        is_eudr_compliant: Whether the area is EUDR-compliant.
        risk_level: Deforestation risk level classification.
        risk_score: Numeric risk score (0 = no risk, 100 = maximum risk).
        baseline_forest_cover_percent: Forest cover at baseline date (%).
        current_forest_cover_percent: Current forest cover (%).
        forest_cover_change_percent: Change in forest cover (%).
        baseline_date: Baseline reference date (ISO YYYY-MM-DD).
        assessment_date: Date of this assessment (ISO YYYY-MM-DD).
        forest_definition: Optional forest definition used for classification.
        deforestation_events: List of deforestation events detected.
        data_sources: List of data sources used in the assessment.
        warnings: List of warning messages or caveats.
    """

    assessment_id: str = Field(
        default="",
        description="Unique identifier for this baseline assessment",
    )
    coordinate_lat: float = Field(
        default=0.0,
        description="Latitude of the assessed point",
    )
    coordinate_lon: float = Field(
        default=0.0,
        description="Longitude of the assessed point",
    )
    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    forest_status: str = Field(
        default="unknown",
        description="Current forest status classification",
    )
    is_eudr_compliant: bool = Field(
        default=True,
        description="Whether the area is EUDR-compliant",
    )
    risk_level: str = Field(
        default="low",
        description="Deforestation risk level classification",
    )
    risk_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Numeric risk score (0 = no risk, 100 = maximum risk)",
    )
    baseline_forest_cover_percent: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Forest cover at baseline date (%)",
    )
    current_forest_cover_percent: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Current forest cover (%)",
    )
    forest_cover_change_percent: float = Field(
        default=0.0,
        description="Change in forest cover (%)",
    )
    baseline_date: str = Field(
        default="",
        description="Baseline reference date (ISO YYYY-MM-DD)",
    )
    assessment_date: str = Field(
        default="",
        description="Date of this assessment (ISO YYYY-MM-DD)",
    )
    forest_definition: Optional[ForestDefinition] = Field(
        None,
        description="Optional forest definition used for classification",
    )
    deforestation_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of deforestation events detected",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="List of data sources used in the assessment",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages or caveats",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.assessment_id:
            self.assessment_id = _new_id("BSA")
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")


class ComplianceReport(BaseModel):
    """EUDR compliance report for a geographic polygon.

    Aggregates satellite analysis, alert data, and risk scoring
    into a formal compliance determination with evidence summary,
    recommendations, and provenance tracking.

    Attributes:
        report_id: Unique identifier for this compliance report.
        polygon_wkt: Well-Known Text representation of the assessed polygon.
        country_iso3: ISO 3166-1 alpha-3 country code.
        compliance_status: Final compliance determination.
        risk_level: Overall deforestation risk level.
        risk_score: Numeric risk score (0 = no risk, 100 = maximum risk).
        total_area_ha: Total area of the polygon in hectares.
        forest_area_ha: Forest area within the polygon in hectares.
        deforested_area_ha: Deforested area within the polygon in hectares.
        total_alerts: Total number of deforestation alerts in the polygon.
        post_cutoff_alerts: Number of alerts after the EUDR cutoff date.
        high_confidence_alerts: Number of high-confidence alerts.
        affected_area_ha: Total area affected by deforestation alerts.
        recommendations: List of recommended actions.
        evidence_summary: Summary of evidence used in the assessment.
        created_at: Report creation timestamp (ISO format).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    report_id: str = Field(
        default="",
        description="Unique identifier for this compliance report",
    )
    polygon_wkt: str = Field(
        default="",
        description="Well-Known Text representation of the assessed polygon",
    )
    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    compliance_status: str = Field(
        default="review_required",
        description="Final compliance determination",
    )
    risk_level: str = Field(
        default="low",
        description="Overall deforestation risk level",
    )
    risk_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Numeric risk score (0 = no risk, 100 = maximum risk)",
    )
    total_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Total area of the polygon in hectares",
    )
    forest_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Forest area within the polygon in hectares",
    )
    deforested_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Deforested area within the polygon in hectares",
    )
    total_alerts: int = Field(
        default=0, ge=0,
        description="Total number of deforestation alerts in the polygon",
    )
    post_cutoff_alerts: int = Field(
        default=0, ge=0,
        description="Number of alerts after the EUDR cutoff date",
    )
    high_confidence_alerts: int = Field(
        default=0, ge=0,
        description="Number of high-confidence alerts",
    )
    affected_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Total area affected by deforestation alerts",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommended actions",
    )
    evidence_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of evidence used in the assessment",
    )
    created_at: str = Field(
        default="",
        description="Report creation timestamp (ISO format)",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.report_id:
            self.report_id = _new_id("RPT")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class MonitoringJob(BaseModel):
    """A recurring deforestation monitoring job for a geographic area.

    Tracks the lifecycle of a scheduled monitoring pipeline including
    the target polygon, schedule frequency, current execution stage,
    and latest results.

    Attributes:
        job_id: Unique identifier for this monitoring job.
        polygon_wkt: Well-Known Text representation of the monitored polygon.
        country_iso3: ISO 3166-1 alpha-3 country code.
        frequency: Monitoring schedule frequency.
        current_stage: Current pipeline stage being executed.
        stages_completed: List of pipeline stages already completed.
        is_running: Whether the job is currently executing.
        started_at: Job start timestamp (ISO format).
        completed_at: Optional job completion timestamp (ISO format).
        last_result: Optional dictionary containing the last execution result.
        error_message: Optional error message if the last execution failed.
    """

    job_id: str = Field(
        default="",
        description="Unique identifier for this monitoring job",
    )
    polygon_wkt: str = Field(
        default="",
        description="Well-Known Text representation of the monitored polygon",
    )
    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    frequency: str = Field(
        default="monthly",
        description="Monitoring schedule frequency",
    )
    current_stage: str = Field(
        default="initialization",
        description="Current pipeline stage being executed",
    )
    stages_completed: List[str] = Field(
        default_factory=list,
        description="List of pipeline stages already completed",
    )
    is_running: bool = Field(
        default=False,
        description="Whether the job is currently executing",
    )
    started_at: str = Field(
        default="",
        description="Job start timestamp (ISO format)",
    )
    completed_at: Optional[str] = Field(
        None,
        description="Optional job completion timestamp (ISO format)",
    )
    last_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional dictionary containing the last execution result",
    )
    error_message: Optional[str] = Field(
        None,
        description="Optional error message if the last execution failed",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.job_id:
            self.job_id = _new_id("MON")
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat()


class PipelineResult(BaseModel):
    """Result of a single pipeline stage execution.

    Records the stage, status, result data, and performance metrics
    for one step of the deforestation monitoring pipeline.

    Attributes:
        pipeline_id: Unique identifier for this pipeline result.
        job_id: Identifier of the parent monitoring job.
        stage: Pipeline stage that produced this result.
        status: Execution status (completed, failed, etc.).
        result_data: Dictionary containing stage-specific result data.
        duration_seconds: Stage execution duration in seconds.
        created_at: Result creation timestamp (ISO format).
    """

    pipeline_id: str = Field(
        default="",
        description="Unique identifier for this pipeline result",
    )
    job_id: str = Field(
        default="",
        description="Identifier of the parent monitoring job",
    )
    stage: str = Field(
        default="initialization",
        description="Pipeline stage that produced this result",
    )
    status: str = Field(
        default="completed",
        description="Execution status",
    )
    result_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing stage-specific result data",
    )
    duration_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Stage execution duration in seconds",
    )
    created_at: str = Field(
        default="",
        description="Result creation timestamp (ISO format)",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.pipeline_id:
            self.pipeline_id = _new_id("PIP")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class AlertAggregation(BaseModel):
    """Aggregated deforestation alert statistics for a geographic area.

    Summarises alert counts by source and severity, total affected area,
    and critical alert presence within a date range and polygon.

    Attributes:
        aggregation_id: Unique identifier for this aggregation.
        polygon_wkt: Well-Known Text representation of the query polygon.
        date_range_start: Start of the aggregation date range (ISO YYYY-MM-DD).
        date_range_end: End of the aggregation date range (ISO YYYY-MM-DD).
        total_alerts: Total number of alerts in the area and period.
        alerts_by_source: Alert count breakdown by source.
        alerts_by_severity: Alert count breakdown by severity.
        total_affected_area_ha: Total area affected by alerts in hectares.
        has_critical: Whether any critical-severity alerts are present.
        high_confidence_count: Number of high-confidence alerts.
    """

    aggregation_id: str = Field(
        default="",
        description="Unique identifier for this aggregation",
    )
    polygon_wkt: str = Field(
        default="",
        description="Well-Known Text representation of the query polygon",
    )
    date_range_start: str = Field(
        default="",
        description="Start of the aggregation date range (ISO YYYY-MM-DD)",
    )
    date_range_end: str = Field(
        default="",
        description="End of the aggregation date range (ISO YYYY-MM-DD)",
    )
    total_alerts: int = Field(
        default=0, ge=0,
        description="Total number of alerts in the area and period",
    )
    alerts_by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert count breakdown by source",
    )
    alerts_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert count breakdown by severity",
    )
    total_affected_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Total area affected by alerts in hectares",
    )
    has_critical: bool = Field(
        default=False,
        description="Whether any critical-severity alerts are present",
    )
    high_confidence_count: int = Field(
        default=0, ge=0,
        description="Number of high-confidence alerts",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.aggregation_id:
            self.aggregation_id = _new_id("AGG")


class TrendAnalysis(BaseModel):
    """Vegetation trend analysis result over a multi-year period.

    Summarises pixel-level NDVI trend slopes to quantify the rate
    and extent of forest decline in a monitored area.

    Attributes:
        pixel_count: Total number of pixels analysed.
        declining_count: Number of pixels with declining NDVI trend.
        declining_area_ha: Area with declining trends in hectares.
        mean_slope: Mean NDVI trend slope across all pixels.
        min_slope: Minimum (most negative) NDVI trend slope.
        max_slope: Maximum (most positive) NDVI trend slope.
        analysis_period_years: Length of the analysis period in years.
    """

    pixel_count: int = Field(
        default=0, ge=0,
        description="Total number of pixels analysed",
    )
    declining_count: int = Field(
        default=0, ge=0,
        description="Number of pixels with declining NDVI trend",
    )
    declining_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Area with declining trends in hectares",
    )
    mean_slope: float = Field(
        default=0.0,
        description="Mean NDVI trend slope across all pixels",
    )
    min_slope: float = Field(
        default=0.0,
        description="Minimum (most negative) NDVI trend slope",
    )
    max_slope: float = Field(
        default=0.0,
        description="Maximum (most positive) NDVI trend slope",
    )
    analysis_period_years: float = Field(
        default=0.0, ge=0.0,
        description="Length of the analysis period in years",
    )

    model_config = ConfigDict(from_attributes=True)


class DeforestationStatistics(BaseModel):
    """Aggregated operational statistics for the deforestation satellite service.

    Provides high-level metrics for monitoring overall service health,
    throughput, and compliance performance.

    Attributes:
        total_scenes: Total number of satellite scenes processed.
        total_assessments: Total number of baseline assessments completed.
        total_alerts: Total number of deforestation alerts processed.
        total_reports: Total number of compliance reports generated.
        total_monitoring_jobs: Total number of monitoring jobs created.
        active_jobs: Number of currently active monitoring jobs.
        forest_area_monitored_ha: Total forest area under monitoring.
        compliance_rate_percent: Percentage of assessments that are compliant.
    """

    total_scenes: int = Field(
        default=0, ge=0,
        description="Total number of satellite scenes processed",
    )
    total_assessments: int = Field(
        default=0, ge=0,
        description="Total number of baseline assessments completed",
    )
    total_alerts: int = Field(
        default=0, ge=0,
        description="Total number of deforestation alerts processed",
    )
    total_reports: int = Field(
        default=0, ge=0,
        description="Total number of compliance reports generated",
    )
    total_monitoring_jobs: int = Field(
        default=0, ge=0,
        description="Total number of monitoring jobs created",
    )
    active_jobs: int = Field(
        default=0, ge=0,
        description="Number of currently active monitoring jobs",
    )
    forest_area_monitored_ha: float = Field(
        default=0.0, ge=0.0,
        description="Total forest area under monitoring in hectares",
    )
    compliance_rate_percent: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of assessments that are compliant",
    )

    model_config = ConfigDict(from_attributes=True)


class CountryRiskProfile(BaseModel):
    """Country-level deforestation risk profile.

    Contains the base risk adjustment factor and list of high-risk
    commodities for a given country, used to modulate risk scoring
    in baseline assessments.

    Attributes:
        country_iso3: ISO 3166-1 alpha-3 country code.
        country_name: Full country name.
        risk_adjustment: Risk score adjustment factor (0.0-2.0).
        high_risk_commodities: List of commodity names with elevated
            deforestation risk in this country.
    """

    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    country_name: str = Field(
        default="",
        description="Full country name",
    )
    risk_adjustment: float = Field(
        default=1.0, ge=0.0,
        description="Risk score adjustment factor",
    )
    high_risk_commodities: List[str] = Field(
        default_factory=list,
        description="List of commodity names with elevated deforestation risk",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Request Models
# =============================================================================


class AcquireSatelliteRequest(BaseModel):
    """Request body for acquiring satellite imagery for a polygon.

    Attributes:
        polygon_coordinates: List of [longitude, latitude] coordinate pairs
            defining the area of interest polygon.
        satellite: Optional satellite source override.
        start_date: Start of the acquisition date range (ISO YYYY-MM-DD).
        end_date: End of the acquisition date range (ISO YYYY-MM-DD).
        max_cloud_cover: Optional maximum cloud cover percentage override.
    """

    polygon_coordinates: List[List[float]] = Field(
        default_factory=list,
        description="List of [longitude, latitude] coordinate pairs",
    )
    satellite: Optional[str] = Field(
        None,
        description="Optional satellite source override",
    )
    start_date: str = Field(
        default="",
        description="Start of the acquisition date range (ISO YYYY-MM-DD)",
    )
    end_date: str = Field(
        default="",
        description="End of the acquisition date range (ISO YYYY-MM-DD)",
    )
    max_cloud_cover: Optional[int] = Field(
        None, ge=0, le=100,
        description="Optional maximum cloud cover percentage override",
    )

    model_config = ConfigDict(extra="forbid")


class DetectChangeRequest(BaseModel):
    """Request body for detecting vegetation change between two temporal windows.

    Attributes:
        polygon_coordinates: List of [longitude, latitude] coordinate pairs
            defining the area of interest polygon.
        pre_start_date: Start of the pre-change date range (ISO YYYY-MM-DD).
        pre_end_date: End of the pre-change date range (ISO YYYY-MM-DD).
        post_start_date: Start of the post-change date range (ISO YYYY-MM-DD).
        post_end_date: End of the post-change date range (ISO YYYY-MM-DD).
        satellite: Optional satellite source override.
    """

    polygon_coordinates: List[List[float]] = Field(
        default_factory=list,
        description="List of [longitude, latitude] coordinate pairs",
    )
    pre_start_date: str = Field(
        default="",
        description="Start of the pre-change date range (ISO YYYY-MM-DD)",
    )
    pre_end_date: str = Field(
        default="",
        description="End of the pre-change date range (ISO YYYY-MM-DD)",
    )
    post_start_date: str = Field(
        default="",
        description="Start of the post-change date range (ISO YYYY-MM-DD)",
    )
    post_end_date: str = Field(
        default="",
        description="End of the post-change date range (ISO YYYY-MM-DD)",
    )
    satellite: Optional[str] = Field(
        None,
        description="Optional satellite source override",
    )

    model_config = ConfigDict(extra="forbid")


class CheckBaselineRequest(BaseModel):
    """Request body for checking EUDR baseline at a single coordinate.

    Attributes:
        latitude: Latitude of the point to assess.
        longitude: Longitude of the point to assess.
        country_iso3: ISO 3166-1 alpha-3 country code.
        observation_date: Optional observation date override (ISO YYYY-MM-DD).
    """

    latitude: float = Field(
        default=0.0,
        description="Latitude of the point to assess",
    )
    longitude: float = Field(
        default=0.0,
        description="Longitude of the point to assess",
    )
    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    observation_date: Optional[str] = Field(
        None,
        description="Optional observation date override (ISO YYYY-MM-DD)",
    )

    model_config = ConfigDict(extra="forbid")


class CheckBaselinePolygonRequest(BaseModel):
    """Request body for checking EUDR baseline across a polygon area.

    Attributes:
        polygon_coordinates: List of [longitude, latitude] coordinate pairs
            defining the area of interest polygon.
        country_iso3: ISO 3166-1 alpha-3 country code.
        observation_date: Optional observation date override (ISO YYYY-MM-DD).
        sample_points: Number of sample points for grid-based assessment.
    """

    polygon_coordinates: List[List[float]] = Field(
        default_factory=list,
        description="List of [longitude, latitude] coordinate pairs",
    )
    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    observation_date: Optional[str] = Field(
        None,
        description="Optional observation date override (ISO YYYY-MM-DD)",
    )
    sample_points: int = Field(
        default=9, ge=1,
        description="Number of sample points for grid-based assessment",
    )

    model_config = ConfigDict(extra="forbid")


class QueryAlertsRequest(BaseModel):
    """Request body for querying deforestation alerts within a polygon and date range.

    Attributes:
        polygon_coordinates: List of [longitude, latitude] coordinate pairs
            defining the area of interest polygon.
        start_date: Start of the query date range (ISO YYYY-MM-DD).
        end_date: End of the query date range (ISO YYYY-MM-DD).
        sources: Optional list of alert sources to filter by.
        min_confidence: Optional minimum confidence level filter.
    """

    polygon_coordinates: List[List[float]] = Field(
        default_factory=list,
        description="List of [longitude, latitude] coordinate pairs",
    )
    start_date: str = Field(
        default="",
        description="Start of the query date range (ISO YYYY-MM-DD)",
    )
    end_date: str = Field(
        default="",
        description="End of the query date range (ISO YYYY-MM-DD)",
    )
    sources: Optional[List[str]] = Field(
        None,
        description="Optional list of alert sources to filter by",
    )
    min_confidence: Optional[str] = Field(
        None,
        description="Optional minimum confidence level filter",
    )

    model_config = ConfigDict(extra="forbid")


class StartMonitoringRequest(BaseModel):
    """Request body for starting a recurring deforestation monitoring job.

    Attributes:
        polygon_coordinates: List of [longitude, latitude] coordinate pairs
            defining the area to monitor.
        country_iso3: ISO 3166-1 alpha-3 country code.
        frequency: Monitoring schedule frequency (default: monthly).
        satellite: Optional satellite source override.
    """

    polygon_coordinates: List[List[float]] = Field(
        default_factory=list,
        description="List of [longitude, latitude] coordinate pairs",
    )
    country_iso3: str = Field(
        default="",
        description="ISO 3166-1 alpha-3 country code",
    )
    frequency: str = Field(
        default="monthly",
        description="Monitoring schedule frequency",
    )
    satellite: Optional[str] = Field(
        None,
        description="Optional satellite source override",
    )

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
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
]
