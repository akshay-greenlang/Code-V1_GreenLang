# -*- coding: utf-8 -*-
"""
Satellite Monitoring Data Models - AGENT-EUDR-003

Pydantic v2 data models for the Satellite Monitoring Agent covering
multi-source satellite imagery acquisition, spectral index calculation,
baseline snapshot establishment, change detection analysis, multi-source
fusion, continuous monitoring scheduling, alert generation, and EUDR
evidence package assembly for EU Deforestation Regulation (EUDR)
Article 9 compliance.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all satellite monitoring operations.

Enumerations (10):
    - SatelliteSource, SpectralIndex, ForestClassification,
      ChangeClassification, DetectionMethod, AlertSeverity,
      MonitoringInterval, EvidenceFormat, CloudFillMethod,
      AnalysisLevel

Core Models (8):
    - SceneMetadata, SceneBand, SpectralIndexResult,
      BaselineSnapshot, ChangeDetectionResult, ChangePixel,
      DataQualityAssessment, CloudCoverAnalysis

Result Models (4):
    - FusionResult, MonitoringResult, SatelliteAlert,
      EvidencePackage

Request Models (6):
    - SearchScenesRequest, EstablishBaselineRequest,
      DetectChangeRequest, CreateMonitoringRequest,
      GenerateEvidenceRequest, BatchAnalysisRequest

Response Models (4):
    - BatchAnalysisResult, BatchProgress,
      AlertSummary, MonitoringSummary

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, and AGENT-EUDR-002
    Geolocation Verification.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from greenlang.agents.data.eudr_traceability.models import EUDRCommodity
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import AlertSeverity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: Maximum number of plots in a single batch analysis request.
MAX_BATCH_SIZE: int = 10_000

#: Standard Sentinel-2 MSI bands available for analysis.
SENTINEL2_BANDS: Dict[str, str] = {
    "B02": "Blue (490nm)",
    "B03": "Green (560nm)",
    "B04": "Red (665nm)",
    "B05": "Vegetation Red Edge (705nm)",
    "B06": "Vegetation Red Edge (740nm)",
    "B07": "Vegetation Red Edge (783nm)",
    "B08": "NIR (842nm)",
    "B8A": "Narrow NIR (865nm)",
    "B11": "SWIR (1610nm)",
    "B12": "SWIR (2190nm)",
    "SCL": "Scene Classification Layer",
}

#: Standard Landsat 8/9 OLI bands available for analysis.
LANDSAT_BANDS: Dict[str, str] = {
    "B2": "Blue (482nm)",
    "B3": "Green (562nm)",
    "B4": "Red (655nm)",
    "B5": "NIR (865nm)",
    "B6": "SWIR1 (1609nm)",
    "B7": "SWIR2 (2201nm)",
    "QA_PIXEL": "Quality Assessment",
}

#: Sentinel-1 SAR polarization bands for cloud-free monitoring.
SENTINEL1_SAR_BANDS: Dict[str, str] = {
    "VV": "Vertical transmit, Vertical receive",
    "VH": "Vertical transmit, Horizontal receive",
    "VV_VH_RATIO": "VV/VH ratio (forest indicator)",
}

#: Default NDVI thresholds for forest classification.
FOREST_NDVI_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "dense_forest": (0.6, 1.0),
    "forest_woodland": (0.4, 0.6),
    "shrubland": (0.2, 0.4),
    "sparse_vegetation": (0.1, 0.2),
    "non_vegetation": (-1.0, 0.1),
}

#: GFW alert source identifiers.
GFW_ALERT_SOURCES: Dict[str, str] = {
    "glad_l": "GLAD Landsat alerts",
    "glad_s2": "GLAD Sentinel-2 alerts",
    "radd": "RADD radar alerts",
    "integrated": "Integrated deforestation alerts",
}

# =============================================================================
# Enumerations
# =============================================================================

class SatelliteSource(str, Enum):
    """Satellite data source for imagery acquisition.

    Identifies the satellite platform and sensor from which imagery
    is acquired for deforestation monitoring.

    SENTINEL_2: ESA Sentinel-2 MSI multispectral imagery (10m optical).
        5-day revisit at equator. Primary optical source.
    LANDSAT_8: USGS Landsat 8 OLI/TIRS imagery (30m optical).
        16-day revisit. Secondary optical source.
    LANDSAT_9: USGS Landsat 9 OLI-2/TIRS-2 imagery (30m optical).
        16-day revisit, offset from Landsat 8 by 8 days.
    SENTINEL_1_SAR: ESA Sentinel-1 C-band SAR imagery.
        Cloud-penetrating radar for wet-season monitoring.
    GFW_ALERTS: Global Forest Watch pre-computed deforestation alerts
        (GLAD Landsat, GLAD Sentinel-2, RADD radar alerts).
    """

    SENTINEL_2 = "sentinel_2"
    LANDSAT_8 = "landsat_8"
    LANDSAT_9 = "landsat_9"
    SENTINEL_1_SAR = "sentinel_1_sar"
    GFW_ALERTS = "gfw_alerts"

class SpectralIndex(str, Enum):
    """Vegetation spectral index type for change detection analysis.

    Each index has specific strengths for detecting different types
    of forest disturbance and land cover change.

    NDVI: Normalized Difference Vegetation Index.
        (NIR - Red) / (NIR + Red). Primary deforestation indicator.
    EVI: Enhanced Vegetation Index.
        Improved sensitivity in high-biomass regions, reduces
        atmospheric noise. Used in tropical forests.
    NBR: Normalized Burn Ratio.
        (NIR - SWIR) / (NIR + SWIR). Detects burned areas and
        post-fire recovery.
    NDMI: Normalized Difference Moisture Index.
        (NIR - SWIR) / (NIR + SWIR). Detects moisture stress
        and drought-induced canopy changes.
    SAVI: Soil-Adjusted Vegetation Index.
        Modified NDVI that minimizes soil brightness influence.
        Used in areas with sparse vegetation cover.
    """

    NDVI = "ndvi"
    EVI = "evi"
    NBR = "nbr"
    NDMI = "ndmi"
    SAVI = "savi"

class ForestClassification(str, Enum):
    """Forest cover classification based on spectral analysis.

    Classifies the land cover type of a pixel or area based on
    spectral index values, used to determine forest status at
    the EUDR cutoff date.

    DENSE_FOREST: Closed canopy forest (NDVI >= 0.6). Indicates
        primary or mature secondary forest.
    FOREST_WOODLAND: Open canopy forest or woodland (NDVI 0.4-0.6).
        May include managed plantations or degraded forest.
    SHRUBLAND: Shrubland or low vegetation (NDVI 0.2-0.4).
        Transition zone between forest and agriculture.
    SPARSE_VEGETATION: Sparse vegetation cover (NDVI 0.1-0.2).
        Savanna, grassland, or recently cleared land.
    NON_VEGETATION: Non-vegetated surface (NDVI < 0.1).
        Urban, water, bare soil, or rock.
    """

    DENSE_FOREST = "dense_forest"
    FOREST_WOODLAND = "forest_woodland"
    SHRUBLAND = "shrubland"
    SPARSE_VEGETATION = "sparse_vegetation"
    NON_VEGETATION = "non_vegetation"

class ChangeClassification(str, Enum):
    """Classification of detected land cover change between two dates.

    Categorizes the nature of change detected through multi-temporal
    spectral analysis relative to the EUDR cutoff date.

    NO_CHANGE: No significant change detected. Land cover is stable
        between baseline and monitoring period.
    DEFORESTATION: Forest to non-forest conversion detected. NDVI
        decrease exceeds the deforestation threshold (-0.15).
        Automatic EUDR compliance failure if post-cutoff.
    DEGRADATION: Forest quality reduction detected without full
        conversion. NDVI decrease between degradation threshold
        (-0.05) and deforestation threshold (-0.15). Warning flag.
    REFORESTATION: Non-forest to forest conversion detected over
        sustained period (> 5 years). Positive EUDR indicator.
    REGROWTH: Short-term vegetation increase detected. NDVI increase
        exceeds regrowth threshold (0.10). May indicate post-harvest
        regeneration or seasonal recovery.
    """

    NO_CHANGE = "no_change"
    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    REFORESTATION = "reforestation"
    REGROWTH = "regrowth"

class DetectionMethod(str, Enum):
    """Method used for change detection analysis.

    Different detection methods have varying accuracy, speed, and
    sensitivity characteristics.

    NDVI_DIFFERENCING: Simple two-date NDVI differencing. Fast but
        sensitive to seasonal variation and atmospheric effects.
    SPECTRAL_ANGLE: Spectral Angle Mapper comparing multi-band
        signatures between dates. More robust than single-index.
    TIME_SERIES_BREAK: BFAST or similar time series breakpoint
        detection. Most accurate but requires long time series.
    MULTI_SOURCE_FUSION: Weighted fusion of multiple data sources
        (Sentinel-2, Landsat, GFW). Highest reliability.
    SAR_BACKSCATTER: Sentinel-1 SAR backscatter change analysis.
        Cloud-independent, detects structural forest changes.
    """

    NDVI_DIFFERENCING = "ndvi_differencing"
    SPECTRAL_ANGLE = "spectral_angle"
    TIME_SERIES_BREAK = "time_series_break"
    MULTI_SOURCE_FUSION = "multi_source_fusion"
    SAR_BACKSCATTER = "sar_backscatter"

class MonitoringInterval(str, Enum):
    """Scheduling interval for continuous monitoring of a plot.

    WEEKLY: Monitor every 7 days. Used for high-risk plots or
        active deforestation fronts.
    BIWEEKLY: Monitor every 14 days. Standard for moderate-risk
        plots or areas with seasonal cloud cover.
    MONTHLY: Monitor every 30 days. Default for low-risk plots
        with established compliance history.
    QUARTERLY: Monitor every 90 days. Used for verified compliant
        plots in stable regions.
    """

    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class EvidenceFormat(str, Enum):
    """Output format for EUDR evidence packages.

    JSON: Machine-readable JSON format for API integration and
        automated processing.
    PDF: Human-readable PDF report with imagery, charts, and
        narrative for regulatory submission.
    CSV: Tabular data export for spreadsheet analysis and
        integration with ERP/compliance systems.
    EUDR_XML: EU Information System structured XML format for
        direct submission to EUDR compliance portal.
    """

    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EUDR_XML = "eudr_xml"

class CloudFillMethod(str, Enum):
    """Method used to fill cloud-covered gaps in optical imagery.

    TEMPORAL_COMPOSITE: Composite multiple cloud-free observations
        from the same sensor within a time window. Most common.
    SAR_FUSION: Use Sentinel-1 SAR backscatter data to supplement
        cloud-covered optical imagery. Cloud-independent.
    INTERPOLATION: Temporal interpolation between nearest cloud-free
        observations. Assumes linear change rate.
    NEAREST_CLEAR: Use the nearest available cloud-free scene
        regardless of time gap. Simplest but least accurate.
    """

    TEMPORAL_COMPOSITE = "temporal_composite"
    SAR_FUSION = "sar_fusion"
    INTERPOLATION = "interpolation"
    NEAREST_CLEAR = "nearest_clear"

class AnalysisLevel(str, Enum):
    """Depth level for satellite analysis operations.

    Controls the thoroughness and computational intensity of the
    analysis for a given plot.

    QUICK: Single-source NDVI differencing. Fastest execution,
        suitable for initial screening. Target: < 10 seconds.
    STANDARD: Multi-source analysis with cloud gap filling.
        Standard DDS preparation level. Target: < 30 seconds.
    DEEP: Full time series analysis with SAR fusion, seasonal
        adjustment, and multi-source validation. Highest accuracy.
        Target: < 120 seconds.
    """

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

# =============================================================================
# Core Data Models
# =============================================================================

class SceneBand(GreenLangBase):
    """Metadata for a single spectral band within a satellite scene.

    Attributes:
        band_id: Band identifier (e.g., 'B04', 'B08', 'VV').
        band_name: Human-readable band name.
        wavelength_nm: Central wavelength in nanometers (optical only).
        resolution_m: Spatial resolution in meters.
        data_type: Pixel data type (e.g., 'uint16', 'float32').
        nodata_value: Value representing no-data pixels.
    """

    model_config = ConfigDict(from_attributes=True)

    band_id: str = Field(
        ...,
        description="Band identifier (e.g., 'B04', 'B08')",
    )
    band_name: str = Field(
        default="",
        description="Human-readable band name",
    )
    wavelength_nm: Optional[float] = Field(
        None,
        ge=0.0,
        description="Central wavelength in nanometers",
    )
    resolution_m: float = Field(
        default=10.0,
        gt=0.0,
        description="Spatial resolution in meters",
    )
    data_type: str = Field(
        default="uint16",
        description="Pixel data type",
    )
    nodata_value: Optional[float] = Field(
        None,
        description="Value representing no-data pixels",
    )

class SceneMetadata(GreenLangBase):
    """Metadata for a single satellite scene (image acquisition).

    Contains identification, temporal, spatial, and quality metadata
    for a scene retrieved from a satellite data source.

    Attributes:
        scene_id: Unique scene identifier from the data provider.
        source: Satellite data source.
        acquisition_date: Date when the scene was acquired.
        acquisition_datetime: Full UTC timestamp of acquisition.
        cloud_cover_pct: Cloud cover percentage for the scene (0-100).
        sun_elevation_deg: Sun elevation angle in degrees at time
            of acquisition.
        sun_azimuth_deg: Sun azimuth angle in degrees.
        processing_level: Processing level (e.g., 'L2A', 'L2SP').
        tile_id: Tile or path/row identifier.
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        crs: Coordinate Reference System (e.g., 'EPSG:32633').
        bands: List of available spectral bands.
        file_size_bytes: Total file size of the scene in bytes.
        download_url: URL for downloading the scene data.
        thumbnail_url: URL for the scene preview thumbnail.
        metadata_url: URL for the full scene metadata document.
    """

    model_config = ConfigDict(from_attributes=True)

    scene_id: str = Field(
        ...,
        description="Unique scene identifier from data provider",
    )
    source: SatelliteSource = Field(
        ...,
        description="Satellite data source",
    )
    acquisition_date: date = Field(
        ...,
        description="Date when the scene was acquired",
    )
    acquisition_datetime: Optional[datetime] = Field(
        None,
        description="Full UTC timestamp of acquisition",
    )
    cloud_cover_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Cloud cover percentage for the scene (0-100)",
    )
    sun_elevation_deg: Optional[float] = Field(
        None,
        ge=0.0,
        le=90.0,
        description="Sun elevation angle in degrees",
    )
    sun_azimuth_deg: Optional[float] = Field(
        None,
        ge=0.0,
        le=360.0,
        description="Sun azimuth angle in degrees",
    )
    processing_level: str = Field(
        default="",
        description="Processing level (e.g., 'L2A', 'L2SP')",
    )
    tile_id: str = Field(
        default="",
        description="Tile or path/row identifier",
    )
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box (min_lon, min_lat, max_lon, max_lat)",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate Reference System",
    )
    bands: List[SceneBand] = Field(
        default_factory=list,
        description="List of available spectral bands",
    )
    file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Total file size of the scene in bytes",
    )
    download_url: str = Field(
        default="",
        description="URL for downloading the scene data",
    )
    thumbnail_url: str = Field(
        default="",
        description="URL for the scene preview thumbnail",
    )
    metadata_url: str = Field(
        default="",
        description="URL for the full scene metadata document",
    )

class SpectralIndexResult(GreenLangBase):
    """Result of computing a spectral vegetation index for a plot.

    Contains the computed index statistics across all pixels within
    the plot boundary for a single scene.

    Attributes:
        index_type: Type of spectral index calculated.
        source: Satellite data source used.
        scene_id: Scene from which the index was computed.
        calculation_date: Date of the scene used for calculation.
        mean_value: Mean index value across plot pixels.
        median_value: Median index value across plot pixels.
        min_value: Minimum index value within the plot.
        max_value: Maximum index value within the plot.
        std_dev: Standard deviation of index values.
        pixel_count: Number of valid pixels used in calculation.
        cloud_free_pct: Percentage of plot pixels that are cloud-free.
        forest_classification: Forest classification based on index.
        quality_score: Data quality score for this calculation (0-100).
    """

    model_config = ConfigDict(from_attributes=True)

    index_type: SpectralIndex = Field(
        ...,
        description="Type of spectral index calculated",
    )
    source: SatelliteSource = Field(
        ...,
        description="Satellite data source used",
    )
    scene_id: str = Field(
        default="",
        description="Scene from which the index was computed",
    )
    calculation_date: date = Field(
        ...,
        description="Date of the scene used for calculation",
    )
    mean_value: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Mean index value across plot pixels",
    )
    median_value: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Median index value across plot pixels",
    )
    min_value: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Minimum index value within the plot",
    )
    max_value: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Maximum index value within the plot",
    )
    std_dev: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of index values",
    )
    pixel_count: int = Field(
        default=0,
        ge=0,
        description="Number of valid pixels used in calculation",
    )
    cloud_free_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of plot pixels that are cloud-free",
    )
    forest_classification: Optional[ForestClassification] = Field(
        None,
        description="Forest classification based on index value",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Data quality score for this calculation (0-100)",
    )

class BaselineSnapshot(GreenLangBase):
    """Spectral baseline snapshot for a plot at the EUDR cutoff date.

    Captures the vegetation state of a plot around December 31, 2020
    using composited cloud-free imagery, establishing the reference
    point for all subsequent change detection analysis.

    Attributes:
        baseline_id: Unique identifier for this baseline snapshot.
        plot_id: Identifier of the plot this baseline represents.
        cutoff_date: The EUDR cutoff date used as the baseline anchor.
        window_start: Start date of the imagery compositing window.
        window_end: End date of the imagery compositing window.
        scenes_used: Number of scenes composited for the baseline.
        scene_ids: List of scene identifiers used in compositing.
        sources_used: List of satellite sources used.
        ndvi_mean: Mean NDVI value for the baseline composite.
        ndvi_median: Median NDVI value for the baseline composite.
        evi_mean: Mean EVI value if computed.
        forest_classification: Forest classification at cutoff date.
        canopy_cover_pct: Estimated canopy cover percentage.
        cloud_free_pct: Percentage of plot with cloud-free data.
        cloud_fill_method: Method used to fill cloud gaps, if any.
        quality_score: Overall quality score for the baseline (0-100).
        provenance_hash: SHA-256 hash for audit trail.
        established_at: UTC timestamp when baseline was established.
    """

    model_config = ConfigDict(from_attributes=True)

    baseline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this baseline snapshot",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the plot this baseline represents",
    )
    cutoff_date: str = Field(
        default=EUDR_CUTOFF_DATE,
        description="EUDR cutoff date used as baseline anchor",
    )
    window_start: Optional[date] = Field(
        None,
        description="Start date of imagery compositing window",
    )
    window_end: Optional[date] = Field(
        None,
        description="End date of imagery compositing window",
    )
    scenes_used: int = Field(
        default=0,
        ge=0,
        description="Number of scenes composited for baseline",
    )
    scene_ids: List[str] = Field(
        default_factory=list,
        description="List of scene identifiers used in compositing",
    )
    sources_used: List[SatelliteSource] = Field(
        default_factory=list,
        description="List of satellite sources used",
    )
    ndvi_mean: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Mean NDVI value for baseline composite",
    )
    ndvi_median: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Median NDVI value for baseline composite",
    )
    evi_mean: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Mean EVI value if computed",
    )
    forest_classification: Optional[ForestClassification] = Field(
        None,
        description="Forest classification at cutoff date",
    )
    canopy_cover_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Estimated canopy cover percentage",
    )
    cloud_free_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of plot with cloud-free data",
    )
    cloud_fill_method: Optional[CloudFillMethod] = Field(
        None,
        description="Method used to fill cloud gaps",
    )
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall quality score for the baseline (0-100)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )
    established_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when baseline was established",
    )

class ChangePixel(GreenLangBase):
    """A single pixel or pixel cluster where land cover change is detected.

    Represents a spatially-located change event with spectral change
    magnitude and classification.

    Attributes:
        latitude: WGS84 latitude of the change pixel centroid.
        longitude: WGS84 longitude of the change pixel centroid.
        ndvi_before: NDVI value in the baseline/before image.
        ndvi_after: NDVI value in the monitoring/after image.
        ndvi_delta: Change in NDVI (after - before).
        area_hectares: Area of the contiguous change cluster in ha.
        classification: Change classification for this pixel cluster.
        confidence: Confidence score for the change detection (0-1).
    """

    model_config = ConfigDict(from_attributes=True)

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude of change pixel centroid",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude of change pixel centroid",
    )
    ndvi_before: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="NDVI value in baseline/before image",
    )
    ndvi_after: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="NDVI value in monitoring/after image",
    )
    ndvi_delta: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Change in NDVI (after - before)",
    )
    area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Area of contiguous change cluster in hectares",
    )
    classification: ChangeClassification = Field(
        default=ChangeClassification.NO_CHANGE,
        description="Change classification for this pixel cluster",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for change detection (0-1)",
    )

class ChangeDetectionResult(GreenLangBase):
    """Result of change detection analysis for a single plot.

    Contains the outcome of comparing the baseline spectral snapshot
    with current imagery to detect deforestation, degradation, or
    regrowth events.

    Attributes:
        detection_id: Unique identifier for this detection run.
        plot_id: Identifier of the monitored plot.
        baseline_id: Identifier of the baseline snapshot used.
        analysis_level: Depth of analysis performed.
        detection_method: Change detection method used.
        baseline_date: Date of the baseline snapshot.
        monitoring_date: Date of the monitoring imagery.
        overall_classification: Overall change classification.
        ndvi_baseline_mean: Mean NDVI from baseline snapshot.
        ndvi_current_mean: Mean NDVI from current imagery.
        ndvi_delta_mean: Mean NDVI change across the plot.
        changed_area_hectares: Total area of detected change (ha).
        changed_area_pct: Percentage of plot area that changed.
        deforested_area_hectares: Area classified as deforestation (ha).
        degraded_area_hectares: Area classified as degradation (ha).
        regrowth_area_hectares: Area classified as regrowth (ha).
        change_pixels: List of individual change pixel clusters.
        confidence_score: Overall confidence in the detection (0-100).
        is_post_cutoff: Whether change occurred after EUDR cutoff.
        sources_used: Satellite sources used in the analysis.
        seasonal_adjusted: Whether seasonal NDVI normalization applied.
        sar_validated: Whether SAR data was used for validation.
        provenance_hash: SHA-256 hash for audit trail.
        detected_at: UTC timestamp of detection.
        processing_time_ms: Processing time in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    detection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this detection run",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the monitored plot",
    )
    baseline_id: str = Field(
        default="",
        description="Identifier of the baseline snapshot used",
    )
    analysis_level: AnalysisLevel = Field(
        default=AnalysisLevel.STANDARD,
        description="Depth of analysis performed",
    )
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.NDVI_DIFFERENCING,
        description="Change detection method used",
    )
    baseline_date: Optional[date] = Field(
        None,
        description="Date of the baseline snapshot",
    )
    monitoring_date: Optional[date] = Field(
        None,
        description="Date of the monitoring imagery",
    )
    overall_classification: ChangeClassification = Field(
        default=ChangeClassification.NO_CHANGE,
        description="Overall change classification for the plot",
    )
    ndvi_baseline_mean: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Mean NDVI from baseline snapshot",
    )
    ndvi_current_mean: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Mean NDVI from current imagery",
    )
    ndvi_delta_mean: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Mean NDVI change across the plot",
    )
    changed_area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Total area of detected change (hectares)",
    )
    changed_area_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of plot area that changed",
    )
    deforested_area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Area classified as deforestation (hectares)",
    )
    degraded_area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Area classified as degradation (hectares)",
    )
    regrowth_area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Area classified as regrowth (hectares)",
    )
    change_pixels: List[ChangePixel] = Field(
        default_factory=list,
        description="Individual change pixel clusters",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall confidence in the detection (0-100)",
    )
    is_post_cutoff: bool = Field(
        default=False,
        description="Whether change occurred after EUDR cutoff",
    )
    sources_used: List[SatelliteSource] = Field(
        default_factory=list,
        description="Satellite sources used in the analysis",
    )
    seasonal_adjusted: bool = Field(
        default=False,
        description="Whether seasonal NDVI normalization was applied",
    )
    sar_validated: bool = Field(
        default=False,
        description="Whether SAR data was used for validation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )
    detected_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of detection",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )

class DataQualityAssessment(GreenLangBase):
    """Data quality assessment for satellite analysis inputs.

    Evaluates the quality and reliability of the satellite data
    used in a monitoring analysis, including temporal coverage,
    spatial completeness, and source consistency.

    Attributes:
        overall_score: Composite quality score (0-100).
        temporal_coverage_score: Score for temporal data coverage.
        spatial_completeness_score: Score for spatial completeness.
        source_consistency_score: Score for cross-source agreement.
        cloud_impact_score: Score for cloud cover impact (higher=less impact).
        scenes_available: Total scenes available for analysis.
        scenes_used: Number of scenes actually used.
        temporal_gap_days: Largest temporal gap in days between scenes.
        sources_available: Number of satellite sources with data.
        issues: List of data quality issues identified.
    """

    model_config = ConfigDict(from_attributes=True)

    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite quality score (0-100)",
    )
    temporal_coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score for temporal data coverage",
    )
    spatial_completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score for spatial completeness",
    )
    source_consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score for cross-source agreement",
    )
    cloud_impact_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Score for cloud cover impact (higher=less impact)",
    )
    scenes_available: int = Field(
        default=0,
        ge=0,
        description="Total scenes available for analysis",
    )
    scenes_used: int = Field(
        default=0,
        ge=0,
        description="Number of scenes actually used",
    )
    temporal_gap_days: int = Field(
        default=0,
        ge=0,
        description="Largest temporal gap in days between scenes",
    )
    sources_available: int = Field(
        default=0,
        ge=0,
        description="Number of satellite sources with data",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Data quality issues identified",
    )

class CloudCoverAnalysis(GreenLangBase):
    """Cloud cover analysis for a plot across a time window.

    Assesses the impact of cloud cover on data availability and
    identifies the most effective gap-filling strategy.

    Attributes:
        plot_id: Identifier of the analyzed plot.
        analysis_start: Start date of the analysis window.
        analysis_end: End date of the analysis window.
        total_scenes: Total scenes in the time window.
        clear_scenes: Number of scenes below cloud threshold.
        cloudy_scenes: Number of scenes above cloud threshold.
        mean_cloud_cover_pct: Mean cloud cover across all scenes.
        min_cloud_cover_pct: Minimum cloud cover scene.
        max_cloud_cover_pct: Maximum cloud cover scene.
        cloud_free_window_days: Longest consecutive cloud-free period.
        recommended_fill_method: Recommended cloud gap fill method.
        sar_scenes_available: Number of SAR scenes for gap filling.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        description="Identifier of the analyzed plot",
    )
    analysis_start: Optional[date] = Field(
        None,
        description="Start date of the analysis window",
    )
    analysis_end: Optional[date] = Field(
        None,
        description="End date of the analysis window",
    )
    total_scenes: int = Field(
        default=0,
        ge=0,
        description="Total scenes in the time window",
    )
    clear_scenes: int = Field(
        default=0,
        ge=0,
        description="Number of scenes below cloud threshold",
    )
    cloudy_scenes: int = Field(
        default=0,
        ge=0,
        description="Number of scenes above cloud threshold",
    )
    mean_cloud_cover_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Mean cloud cover across all scenes",
    )
    min_cloud_cover_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum cloud cover scene",
    )
    max_cloud_cover_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Maximum cloud cover scene",
    )
    cloud_free_window_days: int = Field(
        default=0,
        ge=0,
        description="Longest consecutive cloud-free period in days",
    )
    recommended_fill_method: Optional[CloudFillMethod] = Field(
        None,
        description="Recommended cloud gap fill method",
    )
    sar_scenes_available: int = Field(
        default=0,
        ge=0,
        description="Number of SAR scenes for gap filling",
    )

# =============================================================================
# Result Models
# =============================================================================

class FusionResult(GreenLangBase):
    """Result of multi-source satellite data fusion analysis.

    Combines results from multiple satellite sources (Sentinel-2,
    Landsat, GFW) using weighted fusion to produce a single high-
    confidence change assessment.

    Attributes:
        fusion_id: Unique identifier for this fusion analysis.
        plot_id: Identifier of the monitored plot.
        sources_fused: List of satellite sources included in fusion.
        source_weights: Weight applied to each source.
        sentinel2_result: Change detection result from Sentinel-2.
        landsat_result: Change detection result from Landsat.
        gfw_alert_count: Number of GFW alerts for this plot.
        gfw_alert_dates: Dates of GFW alerts.
        fused_classification: Fused change classification.
        fused_confidence: Fused confidence score (0-100).
        source_agreement: Whether all sources agree on classification.
        disagreement_details: Details of source disagreements, if any.
        data_quality: Data quality assessment for the fusion.
        provenance_hash: SHA-256 hash for audit trail.
        fused_at: UTC timestamp of fusion completion.
    """

    model_config = ConfigDict(from_attributes=True)

    fusion_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this fusion analysis",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the monitored plot",
    )
    sources_fused: List[SatelliteSource] = Field(
        default_factory=list,
        description="Satellite sources included in fusion",
    )
    source_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weight applied to each source",
    )
    sentinel2_result: Optional[ChangeDetectionResult] = Field(
        None,
        description="Change detection result from Sentinel-2",
    )
    landsat_result: Optional[ChangeDetectionResult] = Field(
        None,
        description="Change detection result from Landsat",
    )
    gfw_alert_count: int = Field(
        default=0,
        ge=0,
        description="Number of GFW alerts for this plot",
    )
    gfw_alert_dates: List[date] = Field(
        default_factory=list,
        description="Dates of GFW alerts",
    )
    fused_classification: ChangeClassification = Field(
        default=ChangeClassification.NO_CHANGE,
        description="Fused change classification",
    )
    fused_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Fused confidence score (0-100)",
    )
    source_agreement: bool = Field(
        default=True,
        description="Whether all sources agree on classification",
    )
    disagreement_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details of source disagreements",
    )
    data_quality: Optional[DataQualityAssessment] = Field(
        None,
        description="Data quality assessment for the fusion",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )
    fused_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of fusion completion",
    )

class MonitoringResult(GreenLangBase):
    """Result of a scheduled monitoring execution for a plot.

    Contains the complete outcome of a single monitoring cycle,
    including imagery acquisition, change detection, fusion, and
    alert generation results.

    Attributes:
        monitoring_id: Unique identifier for this monitoring execution.
        plot_id: Identifier of the monitored plot.
        monitoring_interval: Configured monitoring interval.
        execution_number: Sequential execution count for this plot.
        analysis_level: Depth of analysis performed.
        monitoring_date: Date of this monitoring execution.
        baseline_id: Baseline snapshot used for comparison.
        change_result: Change detection result.
        fusion_result: Multi-source fusion result (if standard/deep).
        alerts_generated: List of alerts generated by this execution.
        data_quality: Data quality assessment.
        cloud_analysis: Cloud cover analysis for this period.
        overall_status: Overall monitoring status (clear/alert/warning).
        compliance_flag: EUDR compliance flag (compliant/non_compliant/
            inconclusive).
        provenance_hash: SHA-256 hash for audit trail.
        executed_at: UTC timestamp of monitoring execution.
        processing_time_ms: Total processing time in milliseconds.
        next_scheduled: Next scheduled monitoring date.
    """

    model_config = ConfigDict(from_attributes=True)

    monitoring_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this monitoring execution",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the monitored plot",
    )
    monitoring_interval: MonitoringInterval = Field(
        default=MonitoringInterval.MONTHLY,
        description="Configured monitoring interval",
    )
    execution_number: int = Field(
        default=1,
        ge=1,
        description="Sequential execution count for this plot",
    )
    analysis_level: AnalysisLevel = Field(
        default=AnalysisLevel.STANDARD,
        description="Depth of analysis performed",
    )
    monitoring_date: Optional[date] = Field(
        None,
        description="Date of this monitoring execution",
    )
    baseline_id: str = Field(
        default="",
        description="Baseline snapshot used for comparison",
    )
    change_result: Optional[ChangeDetectionResult] = Field(
        None,
        description="Change detection result",
    )
    fusion_result: Optional[FusionResult] = Field(
        None,
        description="Multi-source fusion result",
    )
    alerts_generated: List[str] = Field(
        default_factory=list,
        description="Alert IDs generated by this execution",
    )
    data_quality: Optional[DataQualityAssessment] = Field(
        None,
        description="Data quality assessment",
    )
    cloud_analysis: Optional[CloudCoverAnalysis] = Field(
        None,
        description="Cloud cover analysis for this period",
    )
    overall_status: str = Field(
        default="clear",
        description="Overall monitoring status (clear/alert/warning)",
    )
    compliance_flag: str = Field(
        default="compliant",
        description="EUDR compliance flag",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )
    executed_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of monitoring execution",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    next_scheduled: Optional[date] = Field(
        None,
        description="Next scheduled monitoring date",
    )

    @field_validator("overall_status")
    @classmethod
    def validate_overall_status(cls, v: str) -> str:
        """Validate overall status is a recognized value."""
        v = v.lower().strip()
        if v not in ("clear", "alert", "warning", "error"):
            raise ValueError(
                f"overall_status must be 'clear', 'alert', 'warning', "
                f"or 'error', got '{v}'"
            )
        return v

    @field_validator("compliance_flag")
    @classmethod
    def validate_compliance_flag(cls, v: str) -> str:
        """Validate compliance flag is a recognized value."""
        v = v.lower().strip()
        if v not in ("compliant", "non_compliant", "inconclusive"):
            raise ValueError(
                f"compliance_flag must be 'compliant', 'non_compliant', "
                f"or 'inconclusive', got '{v}'"
            )
        return v

class SatelliteAlert(GreenLangBase):
    """A deforestation or degradation alert generated by monitoring.

    Represents a single alert raised when satellite analysis detects
    potential EUDR-relevant land cover change on a monitored plot.

    Attributes:
        alert_id: Unique identifier for this alert.
        plot_id: Identifier of the plot that triggered the alert.
        monitoring_id: Monitoring execution that generated this alert.
        severity: Alert severity classification.
        classification: Type of change detected.
        confidence_score: Confidence in the alert (0-100).
        detection_method: Method that detected the change.
        detected_date: Date when the change was detected in imagery.
        area_affected_hectares: Area affected by the detected change.
        area_affected_pct: Percentage of plot area affected.
        ndvi_delta: NDVI change magnitude.
        sources_confirming: List of sources confirming the alert.
        is_post_cutoff: Whether the change is after EUDR cutoff.
        description: Human-readable alert description.
        recommended_action: Recommended follow-up action.
        acknowledged: Whether the alert has been acknowledged.
        acknowledged_by: User who acknowledged the alert.
        acknowledged_at: UTC timestamp of acknowledgement.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: UTC timestamp of alert creation.
    """

    model_config = ConfigDict(from_attributes=True)

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this alert",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the plot that triggered the alert",
    )
    monitoring_id: str = Field(
        default="",
        description="Monitoring execution that generated this alert",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity classification",
    )
    classification: ChangeClassification = Field(
        default=ChangeClassification.DEFORESTATION,
        description="Type of change detected",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence in the alert (0-100)",
    )
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.MULTI_SOURCE_FUSION,
        description="Method that detected the change",
    )
    detected_date: Optional[date] = Field(
        None,
        description="Date when change was detected in imagery",
    )
    area_affected_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Area affected by detected change (hectares)",
    )
    area_affected_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of plot area affected",
    )
    ndvi_delta: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="NDVI change magnitude",
    )
    sources_confirming: List[SatelliteSource] = Field(
        default_factory=list,
        description="Sources confirming the alert",
    )
    is_post_cutoff: bool = Field(
        default=False,
        description="Whether change is after EUDR cutoff",
    )
    description: str = Field(
        default="",
        description="Human-readable alert description",
    )
    recommended_action: str = Field(
        default="",
        description="Recommended follow-up action",
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether the alert has been acknowledged",
    )
    acknowledged_by: Optional[str] = Field(
        None,
        description="User who acknowledged the alert",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp of acknowledgement",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of alert creation",
    )

class EvidencePackage(GreenLangBase):
    """EUDR evidence package for regulatory submission.

    Assembles all satellite monitoring evidence for a plot into a
    structured package suitable for regulatory submission as part
    of a Due Diligence Statement (DDS).

    Attributes:
        package_id: Unique identifier for this evidence package.
        plot_id: Identifier of the plot this evidence covers.
        operator_id: Operator who owns the plot.
        commodity: EUDR commodity produced on this plot.
        country_code: ISO 3166-1 alpha-2 country code.
        format: Output format of the evidence package.
        baseline: Baseline snapshot for the plot.
        change_detections: List of all change detection results.
        fusion_results: List of all fusion analysis results.
        monitoring_history: List of all monitoring execution results.
        alerts: List of all alerts generated for this plot.
        data_quality: Overall data quality assessment.
        overall_classification: Final change classification.
        compliance_status: EUDR compliance status determination.
        confidence_score: Overall evidence confidence (0-100).
        cutoff_date: EUDR cutoff date reference.
        analysis_period_start: Start of the analysis period.
        analysis_period_end: End of the analysis period.
        sources_used: All satellite sources consulted.
        total_scenes_analyzed: Total number of scenes analyzed.
        provenance_hash: SHA-256 hash of the complete package.
        generated_at: UTC timestamp of package generation.
        valid_until: Date until which this evidence is considered
            current (typically monitoring_date + interval).
    """

    model_config = ConfigDict(from_attributes=True)

    package_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this evidence package",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the plot this evidence covers",
    )
    operator_id: str = Field(
        default="",
        description="Operator who owns the plot",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity produced on this plot",
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
    )
    format: EvidenceFormat = Field(
        default=EvidenceFormat.JSON,
        description="Output format of the evidence package",
    )
    baseline: Optional[BaselineSnapshot] = Field(
        None,
        description="Baseline snapshot for the plot",
    )
    change_detections: List[ChangeDetectionResult] = Field(
        default_factory=list,
        description="All change detection results",
    )
    fusion_results: List[FusionResult] = Field(
        default_factory=list,
        description="All fusion analysis results",
    )
    monitoring_history: List[MonitoringResult] = Field(
        default_factory=list,
        description="All monitoring execution results",
    )
    alerts: List[SatelliteAlert] = Field(
        default_factory=list,
        description="All alerts generated for this plot",
    )
    data_quality: Optional[DataQualityAssessment] = Field(
        None,
        description="Overall data quality assessment",
    )
    overall_classification: ChangeClassification = Field(
        default=ChangeClassification.NO_CHANGE,
        description="Final change classification",
    )
    compliance_status: str = Field(
        default="compliant",
        description="EUDR compliance status determination",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall evidence confidence (0-100)",
    )
    cutoff_date: str = Field(
        default=EUDR_CUTOFF_DATE,
        description="EUDR cutoff date reference",
    )
    analysis_period_start: Optional[date] = Field(
        None,
        description="Start of the analysis period",
    )
    analysis_period_end: Optional[date] = Field(
        None,
        description="End of the analysis period",
    )
    sources_used: List[SatelliteSource] = Field(
        default_factory=list,
        description="All satellite sources consulted",
    )
    total_scenes_analyzed: int = Field(
        default=0,
        ge=0,
        description="Total number of scenes analyzed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the complete package",
    )
    generated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of package generation",
    )
    valid_until: Optional[date] = Field(
        None,
        description="Date until which evidence is considered current",
    )

    @field_validator("compliance_status")
    @classmethod
    def validate_compliance_status(cls, v: str) -> str:
        """Validate compliance status is a recognized value."""
        v = v.lower().strip()
        if v not in ("compliant", "non_compliant", "inconclusive"):
            raise ValueError(
                f"compliance_status must be 'compliant', 'non_compliant', "
                f"or 'inconclusive', got '{v}'"
            )
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate and normalize country code to uppercase."""
        if not v:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

# =============================================================================
# Request Models
# =============================================================================

class SearchScenesRequest(GreenLangBase):
    """Request body for searching satellite scenes covering a plot.

    Attributes:
        plot_id: Unique identifier of the plot.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat).
        sources: Satellite sources to search.
        date_start: Start date for the scene search window.
        date_end: End date for the scene search window.
        max_cloud_cover_pct: Maximum acceptable cloud cover percentage.
        max_results: Maximum number of scenes to return.
    """
    plot_id: str = Field(
        ...,
        description="Unique identifier of the plot",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box (min_lon, min_lat, max_lon, max_lat)",
    )
    sources: List[SatelliteSource] = Field(
        default_factory=lambda: [SatelliteSource.SENTINEL_2],
        description="Satellite sources to search",
    )
    date_start: date = Field(
        ...,
        description="Start date for scene search window",
    )
    date_end: date = Field(
        ...,
        description="End date for scene search window",
    )
    max_cloud_cover_pct: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable cloud cover percentage",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of scenes to return",
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plot_id must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> SearchScenesRequest:
        """Validate date_start is before date_end."""
        if self.date_start > self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) must be <= "
                f"date_end ({self.date_end})"
            )
        return self

class EstablishBaselineRequest(GreenLangBase):
    """Request body for establishing a spectral baseline snapshot.

    Attributes:
        plot_id: Unique identifier of the plot.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        polygon: Optional polygon vertices for area analysis.
        cutoff_date: EUDR cutoff date (default 2020-12-31).
        window_days: Number of days for compositing window.
        sources: Satellite sources to use for baseline.
        commodity: EUDR commodity for context.
        country_code: ISO 3166-1 alpha-2 country code.
        operator_id: Operator ID for provenance tracking.
    """
    plot_id: str = Field(
        ...,
        description="Unique identifier of the plot",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    polygon: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Polygon vertices as (lat, lon) tuples",
    )
    cutoff_date: str = Field(
        default=EUDR_CUTOFF_DATE,
        description="EUDR cutoff date (default 2020-12-31)",
    )
    window_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Number of days for compositing window",
    )
    sources: List[SatelliteSource] = Field(
        default_factory=lambda: [
            SatelliteSource.SENTINEL_2,
            SatelliteSource.LANDSAT_8,
        ],
        description="Satellite sources to use for baseline",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for context",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator ID for provenance tracking",
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plot_id must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

class DetectChangeRequest(GreenLangBase):
    """Request body for running change detection on a plot.

    Attributes:
        plot_id: Unique identifier of the plot.
        baseline_id: Baseline snapshot ID for comparison.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        polygon: Optional polygon vertices for area analysis.
        monitoring_date: Target date for monitoring imagery.
        analysis_level: Depth of analysis to perform.
        detection_method: Change detection method to use.
        sources: Satellite sources to use.
        commodity: EUDR commodity for context.
        country_code: ISO 3166-1 alpha-2 country code.
        operator_id: Operator ID for provenance tracking.
    """
    plot_id: str = Field(
        ...,
        description="Unique identifier of the plot",
    )
    baseline_id: str = Field(
        ...,
        description="Baseline snapshot ID for comparison",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    polygon: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Polygon vertices as (lat, lon) tuples",
    )
    monitoring_date: Optional[date] = Field(
        None,
        description="Target date for monitoring imagery",
    )
    analysis_level: AnalysisLevel = Field(
        default=AnalysisLevel.STANDARD,
        description="Depth of analysis to perform",
    )
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.NDVI_DIFFERENCING,
        description="Change detection method to use",
    )
    sources: List[SatelliteSource] = Field(
        default_factory=lambda: [
            SatelliteSource.SENTINEL_2,
            SatelliteSource.LANDSAT_8,
        ],
        description="Satellite sources to use",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for context",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator ID for provenance tracking",
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plot_id must be non-empty")
        return v

    @field_validator("baseline_id")
    @classmethod
    def validate_baseline_id(cls, v: str) -> str:
        """Validate baseline_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("baseline_id must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

class CreateMonitoringRequest(GreenLangBase):
    """Request body for creating a continuous monitoring schedule.

    Attributes:
        plot_id: Unique identifier of the plot.
        baseline_id: Baseline snapshot ID for comparison.
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        polygon: Optional polygon vertices for area analysis.
        monitoring_interval: Scheduling interval for monitoring.
        analysis_level: Depth of analysis for each execution.
        sources: Satellite sources to use.
        commodity: EUDR commodity produced on this plot.
        country_code: ISO 3166-1 alpha-2 country code.
        operator_id: Operator ID for the monitoring schedule.
        start_date: Date to begin monitoring.
        end_date: Optional end date for monitoring schedule.
    """
    plot_id: str = Field(
        ...,
        description="Unique identifier of the plot",
    )
    baseline_id: str = Field(
        ...,
        description="Baseline snapshot ID for comparison",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    polygon: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Polygon vertices as (lat, lon) tuples",
    )
    monitoring_interval: MonitoringInterval = Field(
        default=MonitoringInterval.MONTHLY,
        description="Scheduling interval for monitoring",
    )
    analysis_level: AnalysisLevel = Field(
        default=AnalysisLevel.STANDARD,
        description="Depth of analysis for each execution",
    )
    sources: List[SatelliteSource] = Field(
        default_factory=lambda: [
            SatelliteSource.SENTINEL_2,
            SatelliteSource.LANDSAT_8,
            SatelliteSource.GFW_ALERTS,
        ],
        description="Satellite sources to use",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity produced on this plot",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    operator_id: str = Field(
        ...,
        description="Operator ID for the monitoring schedule",
    )
    start_date: Optional[date] = Field(
        None,
        description="Date to begin monitoring",
    )
    end_date: Optional[date] = Field(
        None,
        description="Optional end date for monitoring schedule",
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plot_id must be non-empty")
        return v

    @field_validator("baseline_id")
    @classmethod
    def validate_baseline_id(cls, v: str) -> str:
        """Validate baseline_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("baseline_id must be non-empty")
        return v

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> CreateMonitoringRequest:
        """Validate start_date is before end_date if both provided."""
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError(
                f"start_date ({self.start_date}) must be <= "
                f"end_date ({self.end_date})"
            )
        return self

class GenerateEvidenceRequest(GreenLangBase):
    """Request body for generating an EUDR evidence package.

    Attributes:
        plot_id: Unique identifier of the plot.
        operator_id: Operator who owns the plot.
        commodity: EUDR commodity produced on this plot.
        country_code: ISO 3166-1 alpha-2 country code.
        format: Desired output format for the evidence package.
        include_imagery_metadata: Whether to include full imagery
            metadata in the package.
        include_monitoring_history: Whether to include all monitoring
            execution history.
    """
    plot_id: str = Field(
        ...,
        description="Unique identifier of the plot",
    )
    operator_id: str = Field(
        ...,
        description="Operator who owns the plot",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity produced on this plot",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    format: EvidenceFormat = Field(
        default=EvidenceFormat.JSON,
        description="Desired output format",
    )
    include_imagery_metadata: bool = Field(
        default=True,
        description="Whether to include full imagery metadata",
    )
    include_monitoring_history: bool = Field(
        default=True,
        description="Whether to include monitoring execution history",
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("plot_id must be non-empty")
        return v

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

class BatchAnalysisRequest(GreenLangBase):
    """Request body for submitting a batch satellite analysis job.

    Attributes:
        plots: List of individual change detection requests.
        analysis_level: Analysis level to apply to all plots.
        priority_country_codes: Optional list of country codes to
            prioritize (analyzed first in the batch).
        operator_id: Operator ID for the batch job.
    """
    plots: List[DetectChangeRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of change detection requests",
    )
    analysis_level: AnalysisLevel = Field(
        default=AnalysisLevel.STANDARD,
        description="Analysis level for all plots",
    )
    priority_country_codes: List[str] = Field(
        default_factory=list,
        description="Country codes to prioritize",
    )
    operator_id: str = Field(
        ...,
        description="Operator ID for the batch job",
    )

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

# =============================================================================
# Response Models
# =============================================================================

class BatchProgress(GreenLangBase):
    """Real-time progress snapshot for a running batch analysis job.

    Provides current processing status for UI integration via
    WebSocket or Server-Sent Events (SSE).

    Attributes:
        batch_id: Unique identifier of the batch job.
        total_plots: Total number of plots in the batch.
        processed: Number of plots processed so far.
        alerts_generated: Number of alerts generated so far.
        errors: Number of plots that encountered errors.
        pending: Number of plots still pending analysis.
        progress_pct: Completion percentage (0-100).
        estimated_remaining_seconds: Estimated time to completion.
        current_plot_id: ID of the plot currently being analyzed.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        ...,
        description="Unique identifier of the batch job",
    )
    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots in the batch",
    )
    processed: int = Field(
        default=0,
        ge=0,
        description="Plots processed so far",
    )
    alerts_generated: int = Field(
        default=0,
        ge=0,
        description="Alerts generated so far",
    )
    errors: int = Field(
        default=0,
        ge=0,
        description="Plots that encountered errors",
    )
    pending: int = Field(
        default=0,
        ge=0,
        description="Plots still pending",
    )
    progress_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage (0-100)",
    )
    estimated_remaining_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated time to completion in seconds",
    )
    current_plot_id: Optional[str] = Field(
        None,
        description="ID of the plot currently being analyzed",
    )

class BatchAnalysisResult(GreenLangBase):
    """Complete result of a batch satellite analysis job.

    Provides aggregate statistics and individual plot results for
    a completed or partially completed batch analysis.

    Attributes:
        batch_id: Unique identifier of the batch job.
        operator_id: Operator who submitted the batch.
        total_plots: Total number of plots in the batch.
        processed: Number of plots processed.
        no_change_count: Number of plots with no change detected.
        deforestation_count: Number of plots with deforestation.
        degradation_count: Number of plots with degradation.
        regrowth_count: Number of plots with regrowth.
        errors_count: Number of plots that encountered errors.
        analysis_level: Analysis level used for all plots.
        alerts_generated: Total alerts generated across all plots.
        average_confidence: Mean confidence score across plots.
        results: List of individual change detection results.
        alerts: List of all alerts generated.
        started_at: UTC timestamp when batch started.
        completed_at: UTC timestamp when batch completed.
        duration_seconds: Total elapsed time in seconds.
        provenance_hash: SHA-256 hash of the complete batch result.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier of the batch job",
    )
    operator_id: str = Field(
        ...,
        description="Operator who submitted the batch",
    )
    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots in the batch",
    )
    processed: int = Field(
        default=0,
        ge=0,
        description="Plots processed",
    )
    no_change_count: int = Field(
        default=0,
        ge=0,
        description="Plots with no change detected",
    )
    deforestation_count: int = Field(
        default=0,
        ge=0,
        description="Plots with deforestation detected",
    )
    degradation_count: int = Field(
        default=0,
        ge=0,
        description="Plots with degradation detected",
    )
    regrowth_count: int = Field(
        default=0,
        ge=0,
        description="Plots with regrowth detected",
    )
    errors_count: int = Field(
        default=0,
        ge=0,
        description="Plots that encountered errors",
    )
    analysis_level: AnalysisLevel = Field(
        default=AnalysisLevel.STANDARD,
        description="Analysis level used",
    )
    alerts_generated: int = Field(
        default=0,
        ge=0,
        description="Total alerts generated",
    )
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Mean confidence score across plots",
    )
    results: List[ChangeDetectionResult] = Field(
        default_factory=list,
        description="Individual change detection results",
    )
    alerts: List[SatelliteAlert] = Field(
        default_factory=list,
        description="All alerts generated",
    )
    started_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when batch started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when batch completed",
    )
    duration_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total elapsed time in seconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the complete batch result",
    )

class AlertSummary(GreenLangBase):
    """Summary statistics for satellite monitoring alerts.

    Provides aggregated alert counts and classifications for
    dashboard displays and compliance reporting.

    Attributes:
        total_alerts: Total number of active alerts.
        critical_count: Number of critical severity alerts.
        warning_count: Number of warning severity alerts.
        info_count: Number of informational alerts.
        deforestation_alerts: Alerts classified as deforestation.
        degradation_alerts: Alerts classified as degradation.
        unacknowledged_count: Number of unacknowledged alerts.
        alerts_by_country: Alert counts grouped by country code.
        alerts_by_commodity: Alert counts grouped by commodity.
        latest_alert_date: Date of the most recent alert.
    """

    model_config = ConfigDict(from_attributes=True)

    total_alerts: int = Field(
        default=0,
        ge=0,
        description="Total number of active alerts",
    )
    critical_count: int = Field(
        default=0,
        ge=0,
        description="Number of critical severity alerts",
    )
    warning_count: int = Field(
        default=0,
        ge=0,
        description="Number of warning severity alerts",
    )
    info_count: int = Field(
        default=0,
        ge=0,
        description="Number of informational alerts",
    )
    deforestation_alerts: int = Field(
        default=0,
        ge=0,
        description="Alerts classified as deforestation",
    )
    degradation_alerts: int = Field(
        default=0,
        ge=0,
        description="Alerts classified as degradation",
    )
    unacknowledged_count: int = Field(
        default=0,
        ge=0,
        description="Number of unacknowledged alerts",
    )
    alerts_by_country: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert counts grouped by country code",
    )
    alerts_by_commodity: Dict[str, int] = Field(
        default_factory=dict,
        description="Alert counts grouped by commodity",
    )
    latest_alert_date: Optional[date] = Field(
        None,
        description="Date of the most recent alert",
    )

class MonitoringSummary(GreenLangBase):
    """Summary statistics for active monitoring schedules.

    Provides aggregated monitoring statistics for dashboard
    displays and operational oversight.

    Attributes:
        total_monitored_plots: Total plots under active monitoring.
        plots_by_interval: Plot counts grouped by monitoring interval.
        plots_by_status: Plot counts grouped by compliance status.
        plots_by_commodity: Plot counts grouped by commodity.
        plots_by_country: Plot counts grouped by country code.
        total_executions: Total monitoring executions performed.
        average_confidence: Mean detection confidence across plots.
        last_execution_date: Date of the most recent execution.
        next_scheduled_date: Date of the next scheduled execution.
        total_area_monitored_hectares: Total area under monitoring.
    """

    model_config = ConfigDict(from_attributes=True)

    total_monitored_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots under active monitoring",
    )
    plots_by_interval: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot counts by monitoring interval",
    )
    plots_by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot counts by compliance status",
    )
    plots_by_commodity: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot counts by commodity",
    )
    plots_by_country: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot counts by country code",
    )
    total_executions: int = Field(
        default=0,
        ge=0,
        description="Total monitoring executions performed",
    )
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Mean detection confidence across plots",
    )
    last_execution_date: Optional[date] = Field(
        None,
        description="Date of the most recent execution",
    )
    next_scheduled_date: Optional[date] = Field(
        None,
        description="Date of the next scheduled execution",
    )
    total_area_monitored_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Total area under monitoring (hectares)",
    )

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "SENTINEL2_BANDS",
    "LANDSAT_BANDS",
    "SENTINEL1_SAR_BANDS",
    "FOREST_NDVI_THRESHOLDS",
    "GFW_ALERT_SOURCES",
    # Re-export for convenience
    "EUDRCommodity",
    # Enumerations
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
    # Core models
    "SceneMetadata",
    "SceneBand",
    "SpectralIndexResult",
    "BaselineSnapshot",
    "ChangeDetectionResult",
    "ChangePixel",
    "DataQualityAssessment",
    "CloudCoverAnalysis",
    # Result models
    "FusionResult",
    "MonitoringResult",
    "SatelliteAlert",
    "EvidencePackage",
    # Request models
    "SearchScenesRequest",
    "EstablishBaselineRequest",
    "DetectChangeRequest",
    "CreateMonitoringRequest",
    "GenerateEvidenceRequest",
    "BatchAnalysisRequest",
    # Response models
    "BatchAnalysisResult",
    "BatchProgress",
    "AlertSummary",
    "MonitoringSummary",
]
