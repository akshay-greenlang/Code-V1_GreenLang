# -*- coding: utf-8 -*-
"""
Deforestation Alert System Data Models - AGENT-EUDR-020

Pydantic v2 data models for the Deforestation Alert System Agent covering
multi-source satellite change detection (Sentinel-2, Landsat, GLAD, Hansen
GFC, RADD), deforestation alert generation with deduplication, five-tier
severity classification with weighted scoring, spatial buffer zone monitoring,
EUDR cutoff date verification (31 December 2020), historical baseline
comparison (2018-2020 reference period), alert workflow management with SLA
tracking, and compliance impact assessment mapping deforestation events to
supply chain disruption and market restrictions.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all deforestation alert operations per
EU 2023/1115 Articles 2, 9, 10, 11, and 31.

Enumerations (12):
    - SatelliteSource, ChangeType, AlertSeverity, AlertStatus,
      BufferType, CutoffResult, ComplianceOutcome, WorkflowAction,
      EUDRCommodity, SpectralIndex, EvidenceQuality, RemediationAction

Core Models (12):
    - SatelliteDetection, DeforestationAlert, SeverityScore,
      SpatialBuffer, BufferViolation, CutoffVerification,
      HistoricalBaseline, BaselineComparison, WorkflowState,
      WorkflowTransition, ComplianceImpact, AuditLogEntry

Request Models (8):
    - DetectChangesRequest, GenerateAlertsRequest,
      ClassifySeverityRequest, CheckBufferRequest,
      VerifyCutoffRequest, CompareBaselineRequest,
      TransitionWorkflowRequest, AssessComplianceRequest

Response Models (8):
    - DetectChangesResponse, GenerateAlertsResponse,
      ClassifySeverityResponse, CheckBufferResponse,
      VerifyCutoffResponse, CompareBaselineResponse,
      TransitionWorkflowResponse, AssessComplianceResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import AlertSeverity, AlertStatus

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: Maximum buffer radius in kilometers.
MAX_BUFFER_RADIUS_KM: int = 50

#: Minimum buffer radius in kilometers.
MIN_BUFFER_RADIUS_KM: int = 1

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 1000

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Supported satellite data sources.
SUPPORTED_SATELLITE_SOURCES: List[str] = [
    "sentinel2",
    "landsat8",
    "landsat9",
    "glad",
    "hansen_gfc",
    "radd",
    "planet",
    "custom",
]

#: Supported spectral vegetation indices.
SUPPORTED_SPECTRAL_INDICES: List[str] = [
    "ndvi",
    "evi",
    "nbr",
    "ndmi",
    "savi",
]

#: EUDR-regulated commodities per Article 1.
SUPPORTED_COMMODITIES: List[str] = [
    "cattle",
    "cocoa",
    "coffee",
    "palm_oil",
    "rubber",
    "soya",
    "wood",
]

#: Default number of points for buffer geometry resolution.
DEFAULT_BUFFER_RESOLUTION: int = 64

# ---------------------------------------------------------------------------
# Enumerations (12)
# ---------------------------------------------------------------------------

class SatelliteSource(str, Enum):
    """Satellite data sources for deforestation monitoring.

    Identifies the origin of satellite imagery used for change detection.
    Each source has different spatial resolution, revisit period, and
    spectral characteristics affecting detection capability.
    """

    SENTINEL2 = "sentinel2"
    """ESA Sentinel-2 MSI: 10m resolution, 5-day revisit, 13 spectral bands."""

    LANDSAT8 = "landsat8"
    """USGS Landsat 8 OLI: 30m resolution, 16-day revisit, 11 bands."""

    LANDSAT9 = "landsat9"
    """USGS Landsat 9 OLI-2: 30m resolution, 16-day revisit, 11 bands."""

    GLAD = "glad"
    """University of Maryland GLAD alerts: weekly Landsat-based deforestation."""

    HANSEN_GFC = "hansen_gfc"
    """Hansen Global Forest Change: annual tree cover loss from Landsat."""

    RADD = "radd"
    """RADD alerts: Sentinel-1 SAR radar-based deforestation detection."""

    PLANET = "planet"
    """Planet Labs: 3-5m resolution, daily revisit, 4-8 spectral bands."""

    CUSTOM = "custom"
    """Custom or third-party satellite data source."""

class ChangeType(str, Enum):
    """Type of land cover change detected by satellite analysis.

    Classifies the nature of the detected change event based on
    spectral signature analysis and temporal patterns.
    """

    DEFORESTATION = "deforestation"
    """Complete removal of forest cover (tree cover loss > 90%)."""

    DEGRADATION = "degradation"
    """Partial loss of forest canopy (30-90% canopy reduction)."""

    FIRE = "fire"
    """Forest fire or burn scar detected via thermal/NBR analysis."""

    LOGGING = "logging"
    """Selective or clear-cut logging activity detected."""

    CLEARING = "clearing"
    """Land clearing for agricultural or development purposes."""

    REGROWTH = "regrowth"
    """Vegetation regrowth or reforestation (positive change)."""

    NO_CHANGE = "no_change"
    """No significant land cover change detected."""

class BufferType(str, Enum):
    """Geometry type for spatial buffer zones around supply chain plots.

    Determines how the monitoring buffer zone is constructed around
    EUDR-regulated production plots.
    """

    CIRCULAR = "circular"
    """Circular buffer with fixed radius from plot centroid."""

    POLYGON = "polygon"
    """Custom polygon buffer following plot boundary with offset."""

    ADAPTIVE = "adaptive"
    """Adaptive buffer adjusting radius based on local deforestation risk."""

class CutoffResult(str, Enum):
    """EUDR cutoff date verification result.

    Classifies whether a detected deforestation event occurred before
    or after the EUDR cutoff date of 31 December 2020 per Article 2(1).
    """

    PRE_CUTOFF = "pre_cutoff"
    """Deforestation occurred before 31 December 2020 (compliant)."""

    POST_CUTOFF = "post_cutoff"
    """Deforestation occurred after 31 December 2020 (non-compliant)."""

    UNCERTAIN = "uncertain"
    """Insufficient temporal evidence to determine timing."""

    ONGOING = "ongoing"
    """Deforestation spans the cutoff date (ongoing event)."""

class ComplianceOutcome(str, Enum):
    """Compliance assessment outcome for deforestation-affected supply chains.

    Maps the deforestation alert analysis to EUDR compliance status
    determining whether products can continue to be placed on the
    EU market.
    """

    COMPLIANT = "compliant"
    """Supply chain remains EUDR compliant (pre-cutoff or resolved)."""

    NON_COMPLIANT = "non_compliant"
    """Supply chain is non-compliant (post-cutoff deforestation confirmed)."""

    UNDER_REVIEW = "under_review"
    """Compliance status under review pending investigation."""

    REMEDIATION_REQUIRED = "remediation_required"
    """Non-compliance detected, remediation plan required."""

class WorkflowAction(str, Enum):
    """Actions that trigger workflow state transitions.

    Defines the set of actions that can be performed on an alert
    to transition it through the workflow lifecycle.
    """

    TRIAGE = "triage"
    """Initial triage assessment and prioritization."""

    ASSIGN = "assign"
    """Assign alert to an investigator or team."""

    INVESTIGATE = "investigate"
    """Begin or continue investigation."""

    RESOLVE = "resolve"
    """Resolve the alert with findings."""

    ESCALATE = "escalate"
    """Escalate to higher authority or management."""

    CLOSE = "close"
    """Close the alert (final state)."""

    REOPEN = "reopen"
    """Reopen a previously resolved or closed alert."""

class EUDRCommodity(str, Enum):
    """EUDR-regulated forest-risk commodities per Article 1.

    The seven commodity categories subject to EUDR due diligence
    requirements, including their derived products.
    """

    CATTLE = "cattle"
    """Cattle and bovine products (leather, beef, dairy)."""

    COCOA = "cocoa"
    """Cocoa beans and derived products (chocolate, cocoa butter)."""

    COFFEE = "coffee"
    """Coffee beans and derived products (roasted, instant)."""

    PALM_OIL = "palm_oil"
    """Oil palm and derived products (palm oil, palm kernel oil)."""

    RUBBER = "rubber"
    """Natural rubber and derived products (tires, latex)."""

    SOYA = "soya"
    """Soybean and derived products (soy meal, soy oil)."""

    WOOD = "wood"
    """Wood and derived products (timber, pulp, paper, furniture)."""

class SpectralIndex(str, Enum):
    """Spectral vegetation indices used for change detection.

    Mathematical combinations of satellite spectral bands that
    highlight vegetation health, moisture content, and burn severity.
    """

    NDVI = "ndvi"
    """Normalized Difference Vegetation Index: (NIR-Red)/(NIR+Red)."""

    EVI = "evi"
    """Enhanced Vegetation Index: improved sensitivity in dense vegetation."""

    NBR = "nbr"
    """Normalized Burn Ratio: (NIR-SWIR)/(NIR+SWIR) for fire detection."""

    NDMI = "ndmi"
    """Normalized Difference Moisture Index: vegetation water content."""

    SAVI = "savi"
    """Soil-Adjusted Vegetation Index: reduces soil background effects."""

class EvidenceQuality(str, Enum):
    """Quality rating for temporal evidence in cutoff date verification.

    Assesses the reliability of evidence used to determine whether
    deforestation occurred before or after the EUDR cutoff date.
    """

    HIGH = "high"
    """High quality: cloud-free imagery, multiple sources, sub-monthly."""

    MEDIUM = "medium"
    """Medium quality: some cloud interference, single source, monthly."""

    LOW = "low"
    """Low quality: significant cloud cover, coarse temporal resolution."""

    INSUFFICIENT = "insufficient"
    """Insufficient: cannot reliably determine temporal placement."""

class RemediationAction(str, Enum):
    """Remediation actions for deforestation-related compliance failures.

    Prescribed corrective actions when deforestation is confirmed to
    affect EUDR-regulated supply chain operations.
    """

    SUPPLIER_AUDIT = "supplier_audit"
    """Conduct on-site audit of affected supplier operations."""

    PLOT_EXCLUSION = "plot_exclusion"
    """Exclude affected plot from sourcing until compliance restored."""

    ALTERNATIVE_SOURCING = "alternative_sourcing"
    """Switch to alternative compliant sourcing for affected products."""

    ENHANCED_MONITORING = "enhanced_monitoring"
    """Increase satellite monitoring frequency for affected area."""

    PRODUCT_WITHDRAWAL = "product_withdrawal"
    """Withdraw affected products from EU market per Article 10."""

# ---------------------------------------------------------------------------
# Core Models (12)
# ---------------------------------------------------------------------------

class SatelliteDetection(GreenLangBase):
    """Satellite-based deforestation change detection event.

    Represents a single change detection result from satellite imagery
    analysis, including the source sensor, geographic location, area
    affected, spectral index values, and detection confidence. This is
    the primary input to the alert generation pipeline.

    Attributes:
        detection_id: Unique identifier for this detection event.
        source: Satellite data source that produced the detection.
        timestamp: UTC timestamp when the satellite image was acquired.
        latitude: Detection center latitude (WGS84, decimal degrees).
        longitude: Detection center longitude (WGS84, decimal degrees).
        geometry_wkt: Well-Known Text representation of detection boundary.
        area_ha: Affected area in hectares.
        change_type: Type of land cover change detected.
        confidence: Detection confidence score (0.0 to 1.0).
        spectral_indices: Spectral index values at detection location.
        cloud_cover_pct: Cloud cover percentage in the scene.
        resolution_m: Spatial resolution in meters.
        tile_id: Satellite tile/scene identifier.
        band_values: Raw spectral band values (optional).
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "detection_id": "det-sentinel2-2025-001",
                    "source": "sentinel2",
                    "latitude": "-3.1234",
                    "longitude": "28.5678",
                    "area_ha": "12.5",
                    "change_type": "deforestation",
                    "confidence": "0.92",
                }
            ]
        },
    )

    detection_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique identifier for this detection event",
    )
    source: SatelliteSource = Field(
        ...,
        description="Satellite data source that produced the detection",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the satellite image was acquired",
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Detection center latitude (WGS84, decimal degrees)",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Detection center longitude (WGS84, decimal degrees)",
    )
    geometry_wkt: Optional[str] = Field(
        None,
        description="Well-Known Text representation of detection boundary",
    )
    area_ha: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Affected area in hectares",
    )
    change_type: ChangeType = Field(
        ...,
        description="Type of land cover change detected",
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Detection confidence score (0.0 to 1.0)",
    )
    spectral_indices: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Spectral index values at detection location",
    )
    cloud_cover_pct: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Cloud cover percentage in the scene",
    )
    resolution_m: Optional[int] = Field(
        None,
        ge=1,
        description="Spatial resolution in meters",
    )
    tile_id: Optional[str] = Field(
        None,
        description="Satellite tile/scene identifier",
    )
    band_values: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Raw spectral band values",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp (UTC)",
    )

class DeforestationAlert(GreenLangBase):
    """Deforestation alert generated from satellite detection analysis.

    Represents an actionable deforestation alert linked to one or more
    satellite detections, with severity classification, geographic
    context, and supply chain plot association. Alerts progress through
    a workflow lifecycle from PENDING through investigation to resolution.

    Attributes:
        alert_id: Unique alert identifier (UUID).
        detection_id: Source satellite detection identifier.
        severity: Alert severity classification.
        status: Current workflow status.
        title: Brief human-readable alert title.
        description: Detailed alert description.
        area_ha: Total affected area in hectares.
        latitude: Alert center latitude (WGS84).
        longitude: Alert center longitude (WGS84).
        country_code: ISO 3166-1 alpha-2 country code.
        affected_plots: List of affected supply chain plot identifiers.
        affected_commodities: EUDR commodities affected by this alert.
        proximity_km: Distance to nearest supply chain plot (km).
        is_post_cutoff: Whether deforestation occurred after EUDR cutoff.
        detection_sources: Satellite sources that contributed to detection.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Alert creation timestamp (UTC).
        updated_at: Alert last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    alert_id: str = Field(
        default_factory=_new_uuid,
        description="Unique alert identifier (UUID)",
    )
    detection_id: str = Field(
        ...,
        description="Source satellite detection identifier",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity classification",
    )
    status: AlertStatus = Field(
        AlertStatus.PENDING,
        description="Current workflow status",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Brief human-readable alert title",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Detailed alert description",
    )
    area_ha: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Total affected area in hectares",
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Alert center latitude (WGS84, decimal degrees)",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Alert center longitude (WGS84, decimal degrees)",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    affected_plots: List[str] = Field(
        default_factory=list,
        description="List of affected supply chain plot identifiers",
    )
    affected_commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="EUDR commodities affected by this alert",
    )
    proximity_km: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Distance to nearest supply chain plot (km)",
    )
    is_post_cutoff: Optional[bool] = Field(
        None,
        description="Whether deforestation occurred after EUDR cutoff",
    )
    detection_sources: List[SatelliteSource] = Field(
        default_factory=list,
        description="Satellite sources that contributed to detection",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Alert creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Alert last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Ensure country_code is uppercase."""
        if v is not None:
            return v.upper()
        return v

class SeverityScore(GreenLangBase):
    """Weighted severity score for deforestation alert classification.

    Implements the five-dimension weighted scoring system: area affected
    (0.25), deforestation rate (0.20), proximity to supply chain plots
    (0.25), protected area overlay (0.15), and post-cutoff timing (0.15).
    Component scores are combined into a 0-100 total score which maps
    to severity levels.

    Attributes:
        score_id: Unique score identifier (UUID).
        alert_id: Alert identifier being scored.
        component_scores: Individual dimension scores (0-100 each).
        weights: Dimension weights (must sum to 1.0).
        total_score: Weighted total score (0-100).
        severity_level: Derived severity classification.
        contributing_factors: Factors that increased the score.
        aggravating_factors: Factors that applied multipliers.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Score computation timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    score_id: str = Field(
        default_factory=_new_uuid,
        description="Unique score identifier (UUID)",
    )
    alert_id: str = Field(
        ...,
        description="Alert identifier being scored",
    )
    component_scores: Dict[str, Decimal] = Field(
        ...,
        description="Individual dimension scores (0-100 each)",
    )
    weights: Dict[str, Decimal] = Field(
        ...,
        description="Dimension weights (must sum to 1.0)",
    )
    total_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted total score (0-100)",
    )
    severity_level: AlertSeverity = Field(
        ...,
        description="Derived severity classification",
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Factors that increased the score",
    )
    aggravating_factors: List[str] = Field(
        default_factory=list,
        description="Factors that applied multipliers (protected area, post-cutoff)",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Score computation timestamp (UTC)",
    )

class SpatialBuffer(GreenLangBase):
    """Spatial monitoring buffer zone around a supply chain plot.

    Defines a geographic buffer zone used to detect deforestation
    events near EUDR-regulated commodity production plots. Buffers
    can be circular (fixed radius), polygon (boundary offset), or
    adaptive (risk-adjusted radius).

    Attributes:
        buffer_id: Unique buffer identifier (UUID).
        plot_id: Supply chain plot identifier being monitored.
        center_lat: Buffer center latitude (WGS84).
        center_lon: Buffer center longitude (WGS84).
        radius_km: Buffer radius in kilometers.
        buffer_type: Buffer geometry type.
        geometry_wkt: Well-Known Text representation of buffer boundary.
        active: Whether this buffer is actively monitored.
        commodities: EUDR commodities associated with the plot.
        country_code: ISO 3166-1 alpha-2 country code.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Buffer creation timestamp (UTC).
        updated_at: Buffer last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    buffer_id: str = Field(
        default_factory=_new_uuid,
        description="Unique buffer identifier (UUID)",
    )
    plot_id: str = Field(
        ...,
        min_length=1,
        description="Supply chain plot identifier being monitored",
    )
    center_lat: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Buffer center latitude (WGS84, decimal degrees)",
    )
    center_lon: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Buffer center longitude (WGS84, decimal degrees)",
    )
    radius_km: Decimal = Field(
        ...,
        ge=Decimal("0.1"),
        le=Decimal("50"),
        description="Buffer radius in kilometers (0.1-50)",
    )
    buffer_type: BufferType = Field(
        BufferType.CIRCULAR,
        description="Buffer geometry type",
    )
    geometry_wkt: Optional[str] = Field(
        None,
        description="Well-Known Text representation of buffer boundary",
    )
    active: bool = Field(
        True,
        description="Whether this buffer is actively monitored",
    )
    commodities: List[EUDRCommodity] = Field(
        default_factory=list,
        description="EUDR commodities associated with the plot",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Buffer creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="Buffer last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Ensure country_code is uppercase."""
        if v is not None:
            return v.upper()
        return v

class BufferViolation(GreenLangBase):
    """Detection of deforestation within a supply chain plot buffer zone.

    Records when a satellite detection falls within the monitoring
    buffer of a supply chain plot, triggering proximity-based alerting.

    Attributes:
        violation_id: Unique violation identifier (UUID).
        buffer_id: Buffer zone that was violated.
        detection_id: Satellite detection that caused the violation.
        distance_km: Distance from detection to buffer center (km).
        overlap_area_ha: Area of overlap between detection and buffer (ha).
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Violation detection timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    violation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique violation identifier (UUID)",
    )
    buffer_id: str = Field(
        ...,
        description="Buffer zone that was violated",
    )
    detection_id: str = Field(
        ...,
        description="Satellite detection that caused the violation",
    )
    distance_km: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Distance from detection to buffer center (km)",
    )
    overlap_area_ha: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Area of overlap between detection and buffer (ha)",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Violation detection timestamp (UTC)",
    )

class CutoffVerification(GreenLangBase):
    """EUDR cutoff date temporal verification result.

    Verifies whether detected deforestation occurred before or after
    the EUDR cutoff date of 31 December 2020 using multi-source
    temporal evidence analysis with confidence scoring.

    Attributes:
        verification_id: Unique verification identifier (UUID).
        detection_id: Satellite detection being verified.
        cutoff_result: Temporal classification result.
        confidence: Confidence score for the determination (0-1).
        evidence_sources: List of evidence source descriptions.
        earliest_detection_date: Earliest date deforestation was observed.
        latest_clear_date: Latest date area was confirmed forested.
        temporal_analysis: Detailed temporal analysis data.
        evidence_quality: Quality rating of the evidence used.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Verification timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    verification_id: str = Field(
        default_factory=_new_uuid,
        description="Unique verification identifier (UUID)",
    )
    detection_id: str = Field(
        ...,
        description="Satellite detection being verified",
    )
    cutoff_result: CutoffResult = Field(
        ...,
        description="Temporal classification result",
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence score for the determination (0-1)",
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="List of evidence source descriptions",
    )
    earliest_detection_date: Optional[date] = Field(
        None,
        description="Earliest date deforestation was observed",
    )
    latest_clear_date: Optional[date] = Field(
        None,
        description="Latest date area was confirmed forested",
    )
    temporal_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed temporal analysis data",
    )
    evidence_quality: EvidenceQuality = Field(
        EvidenceQuality.MEDIUM,
        description="Quality rating of the evidence used",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Verification timestamp (UTC)",
    )

class HistoricalBaseline(GreenLangBase):
    """Historical reference baseline for a supply chain plot location.

    Establishes the reference forest cover state during the baseline
    period (default 2018-2020) against which current conditions are
    compared to detect deforestation. Requires minimum 3 cloud-free
    observations and 10% canopy cover threshold to classify as forested.

    Attributes:
        baseline_id: Unique baseline identifier (UUID).
        plot_id: Supply chain plot identifier.
        latitude: Plot center latitude (WGS84).
        longitude: Plot center longitude (WGS84).
        baseline_period: Description of the baseline period (e.g. "2018-2020").
        canopy_cover_pct: Average canopy cover during baseline (%).
        forest_area_ha: Forested area during baseline period (ha).
        reference_images: List of reference image identifiers used.
        num_observations: Number of cloud-free observations in baseline.
        established_at: Date baseline was established.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    baseline_id: str = Field(
        default_factory=_new_uuid,
        description="Unique baseline identifier (UUID)",
    )
    plot_id: str = Field(
        ...,
        min_length=1,
        description="Supply chain plot identifier",
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Plot center latitude (WGS84, decimal degrees)",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Plot center longitude (WGS84, decimal degrees)",
    )
    baseline_period: str = Field(
        "2018-2020",
        description="Description of the baseline period",
    )
    canopy_cover_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Average canopy cover during baseline (%)",
    )
    forest_area_ha: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Forested area during baseline period (ha)",
    )
    reference_images: List[str] = Field(
        default_factory=list,
        description="List of reference image identifiers used",
    )
    num_observations: int = Field(
        0,
        ge=0,
        description="Number of cloud-free observations in baseline",
    )
    established_at: datetime = Field(
        default_factory=utcnow,
        description="Date baseline was established",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp (UTC)",
    )

class BaselineComparison(GreenLangBase):
    """Comparison of current conditions against historical baseline.

    Quantifies the change between the historical baseline reference
    state and current satellite observations to detect and measure
    deforestation at a specific location.

    Attributes:
        comparison_id: Unique comparison identifier (UUID).
        baseline_id: Historical baseline being compared against.
        current_date: Date of the current observation.
        canopy_change_pct: Percentage change in canopy cover.
        area_change_ha: Change in forested area (ha, negative = loss).
        change_type: Classified type of change.
        confidence: Confidence score for the comparison (0-1).
        current_canopy_pct: Current canopy cover percentage.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Comparison timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    comparison_id: str = Field(
        default_factory=_new_uuid,
        description="Unique comparison identifier (UUID)",
    )
    baseline_id: str = Field(
        ...,
        description="Historical baseline being compared against",
    )
    current_date: date = Field(
        ...,
        description="Date of the current observation",
    )
    canopy_change_pct: Decimal = Field(
        ...,
        ge=Decimal("-100"),
        le=Decimal("100"),
        description="Percentage change in canopy cover",
    )
    area_change_ha: Decimal = Field(
        ...,
        description="Change in forested area (ha, negative = loss)",
    )
    change_type: ChangeType = Field(
        ...,
        description="Classified type of change",
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence score for the comparison (0-1)",
    )
    current_canopy_pct: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Current canopy cover percentage",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Comparison timestamp (UTC)",
    )

class WorkflowTransition(GreenLangBase):
    """A single state transition in the alert workflow lifecycle.

    Records the change from one workflow status to another, capturing
    the action performed, the actor, and any notes for audit trail.

    Attributes:
        from_status: Previous workflow status.
        to_status: New workflow status.
        action: Action that triggered the transition.
        actor: User or system that performed the action.
        timestamp: Transition timestamp (UTC).
        notes: Optional notes about the transition.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    from_status: AlertStatus = Field(
        ...,
        description="Previous workflow status",
    )
    to_status: AlertStatus = Field(
        ...,
        description="New workflow status",
    )
    action: WorkflowAction = Field(
        ...,
        description="Action that triggered the transition",
    )
    actor: str = Field(
        "system",
        description="User or system that performed the action",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Transition timestamp (UTC)",
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional notes about the transition",
    )

class WorkflowState(GreenLangBase):
    """Current workflow state for a deforestation alert.

    Tracks the alert's position in the investigation workflow including
    assignment, priority, SLA deadline, transition history, and notes.

    Attributes:
        state_id: Unique state identifier (UUID).
        alert_id: Alert identifier being tracked.
        current_status: Current workflow status.
        assigned_to: Assigned investigator or team.
        priority: Priority level (1=highest, 5=lowest).
        sla_deadline: SLA deadline timestamp (UTC).
        escalation_level: Current escalation level (0=none).
        notes: List of workflow notes.
        transitions: History of workflow state transitions.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: State creation timestamp (UTC).
        updated_at: State last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    state_id: str = Field(
        default_factory=_new_uuid,
        description="Unique state identifier (UUID)",
    )
    alert_id: str = Field(
        ...,
        description="Alert identifier being tracked",
    )
    current_status: AlertStatus = Field(
        AlertStatus.PENDING,
        description="Current workflow status",
    )
    assigned_to: Optional[str] = Field(
        None,
        description="Assigned investigator or team",
    )
    priority: int = Field(
        3,
        ge=1,
        le=5,
        description="Priority level (1=highest, 5=lowest)",
    )
    sla_deadline: Optional[datetime] = Field(
        None,
        description="SLA deadline timestamp (UTC)",
    )
    escalation_level: int = Field(
        0,
        ge=0,
        le=5,
        description="Current escalation level (0=none)",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="List of workflow notes",
    )
    transitions: List[WorkflowTransition] = Field(
        default_factory=list,
        description="History of workflow state transitions",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="State creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="State last update timestamp (UTC)",
    )

class ComplianceImpact(GreenLangBase):
    """Compliance impact assessment for a deforestation alert.

    Maps the deforestation alert analysis to supply chain impact,
    compliance outcome, market restriction decisions, required
    remediation actions, and estimated financial exposure.

    Attributes:
        impact_id: Unique impact assessment identifier (UUID).
        alert_id: Alert identifier being assessed.
        affected_suppliers: List of affected supplier identifiers.
        affected_products: List of affected product identifiers.
        compliance_outcome: Overall compliance assessment result.
        market_restriction: Whether market restriction is recommended.
        remediation_actions: Required or recommended remediation actions.
        estimated_financial_impact: Estimated financial exposure (EUR).
        risk_score: Overall compliance risk score (0-100).
        assessment_notes: Assessment notes and justification.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Assessment timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    impact_id: str = Field(
        default_factory=_new_uuid,
        description="Unique impact assessment identifier (UUID)",
    )
    alert_id: str = Field(
        ...,
        description="Alert identifier being assessed",
    )
    affected_suppliers: List[str] = Field(
        default_factory=list,
        description="List of affected supplier identifiers",
    )
    affected_products: List[str] = Field(
        default_factory=list,
        description="List of affected product identifiers",
    )
    compliance_outcome: ComplianceOutcome = Field(
        ...,
        description="Overall compliance assessment result",
    )
    market_restriction: bool = Field(
        False,
        description="Whether market restriction is recommended",
    )
    remediation_actions: List[RemediationAction] = Field(
        default_factory=list,
        description="Required or recommended remediation actions",
    )
    estimated_financial_impact: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Estimated financial exposure in EUR",
    )
    risk_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Overall compliance risk score (0-100)",
    )
    assessment_notes: Optional[str] = Field(
        None,
        max_length=5000,
        description="Assessment notes and justification",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Assessment timestamp (UTC)",
    )

class AuditLogEntry(GreenLangBase):
    """Audit log entry for deforestation alert system operations.

    Captures all significant operations for EUDR Article 31 compliance
    audit trail requirements including who performed what action on
    which entity and when.

    Attributes:
        entry_id: Unique audit entry identifier (UUID).
        operation: Operation performed.
        entity_type: Type of entity affected.
        entity_id: Identifier of the affected entity.
        actor: User or system that performed the operation.
        details: Operation details and context.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Audit entry timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique audit entry identifier (UUID)",
    )
    operation: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Operation performed",
    )
    entity_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of entity affected",
    )
    entity_id: str = Field(
        ...,
        description="Identifier of the affected entity",
    )
    actor: str = Field(
        "system",
        description="User or system that performed the operation",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation details and context",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Audit entry timestamp (UTC)",
    )

# ---------------------------------------------------------------------------
# Request Models (8)
# ---------------------------------------------------------------------------

class DetectChangesRequest(GreenLangBase):
    """Request to perform satellite change detection.

    Triggers change detection analysis on satellite imagery for a
    specified geographic region and time window.

    Attributes:
        region_wkt: Well-Known Text of the analysis region.
        latitude: Center latitude for point-based detection.
        longitude: Center longitude for point-based detection.
        radius_km: Search radius in km for point-based detection.
        start_date: Start date for temporal window.
        end_date: End date for temporal window.
        sources: Satellite sources to use (default: all enabled).
        min_confidence: Minimum confidence threshold override.
        max_cloud_cover_pct: Maximum cloud cover override.
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    region_wkt: Optional[str] = Field(
        None,
        description="Well-Known Text of the analysis region",
    )
    latitude: Optional[Decimal] = Field(
        None,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Center latitude for point-based detection",
    )
    longitude: Optional[Decimal] = Field(
        None,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Center longitude for point-based detection",
    )
    radius_km: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.1"),
        le=Decimal("100"),
        description="Search radius in km for point-based detection",
    )
    start_date: Optional[date] = Field(
        None,
        description="Start date for temporal window",
    )
    end_date: Optional[date] = Field(
        None,
        description="End date for temporal window",
    )
    sources: Optional[List[SatelliteSource]] = Field(
        None,
        description="Satellite sources to use (default: all enabled)",
    )
    min_confidence: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Minimum confidence threshold override",
    )
    max_cloud_cover_pct: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Maximum cloud cover override",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class GenerateAlertsRequest(GreenLangBase):
    """Request to generate deforestation alerts from detections.

    Attributes:
        detections: List of satellite detections to process.
        plot_ids: Optional list of plot IDs to check for proximity.
        dedup_enabled: Whether to apply deduplication (default: True).
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    detections: List[SatelliteDetection] = Field(
        ...,
        min_length=1,
        description="List of satellite detections to process",
    )
    plot_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of plot IDs to check for proximity",
    )
    dedup_enabled: bool = Field(
        True,
        description="Whether to apply deduplication",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class ClassifySeverityRequest(GreenLangBase):
    """Request to classify the severity of a deforestation alert.

    Attributes:
        alert: The deforestation alert to classify.
        nearby_plots: Nearby supply chain plots for proximity scoring.
        is_protected_area: Whether detection is in a protected area.
        is_post_cutoff: Whether detection is after EUDR cutoff date.
        deforestation_rate_ha_per_day: Rate of deforestation (ha/day).
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    alert: DeforestationAlert = Field(
        ...,
        description="The deforestation alert to classify",
    )
    nearby_plots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nearby supply chain plots for proximity scoring",
    )
    is_protected_area: bool = Field(
        False,
        description="Whether detection is in a protected area",
    )
    is_post_cutoff: bool = Field(
        False,
        description="Whether detection is after EUDR cutoff date",
    )
    deforestation_rate_ha_per_day: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Rate of deforestation (ha/day)",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class CheckBufferRequest(GreenLangBase):
    """Request to check for buffer zone violations.

    Attributes:
        detection: Satellite detection to check against buffers.
        buffer_ids: Optional list of specific buffer IDs to check.
        active_only: Only check active buffers (default: True).
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    detection: SatelliteDetection = Field(
        ...,
        description="Satellite detection to check against buffers",
    )
    buffer_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of specific buffer IDs to check",
    )
    active_only: bool = Field(
        True,
        description="Only check active buffers",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class VerifyCutoffRequest(GreenLangBase):
    """Request to verify EUDR cutoff date for a detection.

    Attributes:
        detection_id: Detection identifier to verify.
        latitude: Detection latitude.
        longitude: Detection longitude.
        detection_date: Date of the detection.
        evidence_sources: Available temporal evidence sources.
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    detection_id: str = Field(
        ...,
        description="Detection identifier to verify",
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Detection latitude",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Detection longitude",
    )
    detection_date: date = Field(
        ...,
        description="Date of the detection",
    )
    evidence_sources: Optional[List[str]] = Field(
        None,
        description="Available temporal evidence sources",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class CompareBaselineRequest(GreenLangBase):
    """Request to compare current conditions against historical baseline.

    Attributes:
        baseline_id: Baseline identifier to compare against.
        current_canopy_pct: Current canopy cover percentage.
        current_forest_area_ha: Current forested area (ha).
        observation_date: Date of the current observation.
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    baseline_id: str = Field(
        ...,
        description="Baseline identifier to compare against",
    )
    current_canopy_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Current canopy cover percentage",
    )
    current_forest_area_ha: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Current forested area (ha)",
    )
    observation_date: date = Field(
        ...,
        description="Date of the current observation",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class TransitionWorkflowRequest(GreenLangBase):
    """Request to transition an alert through the workflow.

    Attributes:
        alert_id: Alert identifier to transition.
        action: Workflow action to perform.
        actor: User or system performing the action.
        assigned_to: New assignee (for ASSIGN action).
        notes: Optional transition notes.
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str = Field(
        ...,
        description="Alert identifier to transition",
    )
    action: WorkflowAction = Field(
        ...,
        description="Workflow action to perform",
    )
    actor: str = Field(
        "system",
        description="User or system performing the action",
    )
    assigned_to: Optional[str] = Field(
        None,
        description="New assignee (for ASSIGN action)",
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional transition notes",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

class AssessComplianceRequest(GreenLangBase):
    """Request to assess compliance impact of a deforestation alert.

    Attributes:
        alert_id: Alert identifier to assess.
        alert: The deforestation alert being assessed.
        severity_score: Severity score of the alert.
        cutoff_verification: Cutoff date verification result.
        affected_suppliers: Known affected supplier identifiers.
        affected_products: Known affected product identifiers.
        request_id: Optional client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str = Field(
        ...,
        description="Alert identifier to assess",
    )
    alert: Optional[DeforestationAlert] = Field(
        None,
        description="The deforestation alert being assessed",
    )
    severity_score: Optional[SeverityScore] = Field(
        None,
        description="Severity score of the alert",
    )
    cutoff_verification: Optional[CutoffVerification] = Field(
        None,
        description="Cutoff date verification result",
    )
    affected_suppliers: List[str] = Field(
        default_factory=list,
        description="Known affected supplier identifiers",
    )
    affected_products: List[str] = Field(
        default_factory=list,
        description="Known affected product identifiers",
    )
    request_id: Optional[str] = Field(
        None,
        description="Optional client-provided request identifier",
    )

# ---------------------------------------------------------------------------
# Response Models (8)
# ---------------------------------------------------------------------------

class DetectChangesResponse(GreenLangBase):
    """Response from satellite change detection.

    Attributes:
        detections: List of detected change events.
        total_detections: Total number of detections found.
        sources_queried: Satellite sources that were queried.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    detections: List[SatelliteDetection] = Field(
        default_factory=list,
        description="List of detected change events",
    )
    total_detections: int = Field(
        0,
        ge=0,
        description="Total number of detections found",
    )
    sources_queried: List[SatelliteSource] = Field(
        default_factory=list,
        description="Satellite sources that were queried",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class GenerateAlertsResponse(GreenLangBase):
    """Response from alert generation.

    Attributes:
        alerts: List of generated deforestation alerts.
        total_generated: Total number of alerts generated.
        duplicates_filtered: Number of duplicate detections filtered.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    alerts: List[DeforestationAlert] = Field(
        default_factory=list,
        description="List of generated deforestation alerts",
    )
    total_generated: int = Field(
        0,
        ge=0,
        description="Total number of alerts generated",
    )
    duplicates_filtered: int = Field(
        0,
        ge=0,
        description="Number of duplicate detections filtered",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class ClassifySeverityResponse(GreenLangBase):
    """Response from severity classification.

    Attributes:
        severity_score: Computed severity score with breakdown.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    severity_score: SeverityScore = Field(
        ...,
        description="Computed severity score with breakdown",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class CheckBufferResponse(GreenLangBase):
    """Response from buffer zone violation check.

    Attributes:
        violations: List of buffer zone violations found.
        total_violations: Total number of violations detected.
        buffers_checked: Number of buffer zones checked.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    violations: List[BufferViolation] = Field(
        default_factory=list,
        description="List of buffer zone violations found",
    )
    total_violations: int = Field(
        0,
        ge=0,
        description="Total number of violations detected",
    )
    buffers_checked: int = Field(
        0,
        ge=0,
        description="Number of buffer zones checked",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class VerifyCutoffResponse(GreenLangBase):
    """Response from EUDR cutoff date verification.

    Attributes:
        verification: Cutoff verification result.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    verification: CutoffVerification = Field(
        ...,
        description="Cutoff verification result",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class CompareBaselineResponse(GreenLangBase):
    """Response from historical baseline comparison.

    Attributes:
        comparison: Baseline comparison result.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    comparison: BaselineComparison = Field(
        ...,
        description="Baseline comparison result",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class TransitionWorkflowResponse(GreenLangBase):
    """Response from workflow state transition.

    Attributes:
        workflow_state: Updated workflow state after transition.
        transition: The transition that was executed.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_state: WorkflowState = Field(
        ...,
        description="Updated workflow state after transition",
    )
    transition: WorkflowTransition = Field(
        ...,
        description="The transition that was executed",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )

class AssessComplianceResponse(GreenLangBase):
    """Response from compliance impact assessment.

    Attributes:
        impact: Compliance impact assessment result.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier (echoed).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    impact: ComplianceImpact = Field(
        ...,
        description="Compliance impact assessment result",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier (echoed)",
    )
