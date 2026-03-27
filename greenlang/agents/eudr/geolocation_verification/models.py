# -*- coding: utf-8 -*-
"""
Geolocation Verification Data Models - AGENT-EUDR-002

Pydantic v2 data models for the Geolocation Verification Agent covering
multi-layer coordinate validation, polygon topology verification,
protected area intersection analysis, deforestation cutoff verification,
accuracy scoring, temporal consistency analysis, batch verification
pipelines, and Article 9 compliance reporting for EU Deforestation
Regulation (EUDR) compliance.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all geolocation verification operations.

Enumerations (8):
    - VerificationLevel, VerificationStatus, CoordinateIssueType,
      PolygonIssueType, OverlapSeverity, DeforestationStatus,
      QualityTier, ChangeType

Core Models (6):
    - CoordinateIssue, PolygonIssue, RepairSuggestion,
      ProtectedAreaOverlap, ProtectedAreaProximity, TreeCoverLossEvent

Result Models (7):
    - CoordinateValidationResult, PolygonVerificationResult,
      ProtectedAreaCheckResult, DeforestationVerificationResult,
      GeolocationAccuracyScore, TemporalChangeResult, BoundaryChange

Request Models (5):
    - VerifyCoordinateRequest, VerifyPolygonRequest, VerifyPlotRequest,
      BatchVerificationRequest, ComplianceReportRequest

Response Models (5):
    - PlotVerificationResult, BatchVerificationResult, BatchProgress,
      ComplianceReport, ComplianceSummary

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-001 Supply Chain Mapper.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from greenlang.agents.data.eudr_traceability.models import EUDRCommodity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Default accuracy score weights as specified in PRD Section 6.1 Feature 5.
DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "precision": 0.20,
    "polygon": 0.20,
    "country": 0.15,
    "protected": 0.15,
    "deforestation": 0.15,
    "temporal": 0.15,
}

#: Quality tier thresholds mapping tier to (min_score, max_score).
QUALITY_TIER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "gold": (90.0, 100.0),
    "silver": (70.0, 89.99),
    "bronze": (50.0, 69.99),
    "unverified": (0.0, 49.99),
}

#: Maximum number of plots in a single batch verification request.
MAX_BATCH_SIZE: int = 50_000


# =============================================================================
# Enumerations
# =============================================================================


class VerificationLevel(str, Enum):
    """Verification depth level for plot geolocation checks.

    Controls how many verification engines are invoked for a given plot.

    QUICK: Coordinate validation and polygon topology only. Fastest
        execution, suitable for initial data ingestion screening.
        Target: < 5 seconds per plot.
    STANDARD: Includes QUICK plus protected area screening and country
        boundary verification. Suitable for pre-DDS review.
        Target: < 30 seconds per plot.
    DEEP: Includes STANDARD plus deforestation cutoff verification
        and temporal consistency analysis. Full verification for
        DDS submission readiness. Target: < 120 seconds per plot.
    """

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class VerificationStatus(str, Enum):
    """Overall verification outcome for a plot or verification component.

    PENDING: Verification has not yet been performed or is in progress.
    PASSED: All checks passed; plot meets EUDR Article 9 requirements.
    FAILED: One or more critical checks failed; plot does not meet
        EUDR requirements and requires remediation.
    WARNING: All checks passed but with minor issues that should be
        reviewed (e.g., low coordinate precision, marginal overlap).
    """

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class CoordinateIssueType(str, Enum):
    """Classification of coordinate validation issues.

    Each issue type maps to a specific validation check performed
    by the Coordinate Validation Engine (Feature 1).

    OUT_OF_BOUNDS: Coordinate exceeds WGS84 valid ranges
        (lat not in [-90, 90] or lon not in [-180, 180]).
    LOW_PRECISION: Coordinate has fewer decimal places than the
        minimum threshold (default: 5 decimal places).
    LIKELY_TRANSPOSED: Latitude and longitude values appear to be
        swapped (e.g., latitude > 90 but valid as longitude).
    COUNTRY_MISMATCH: Coordinate falls outside the declared
        country boundaries.
    ON_WATER: Coordinate falls in an ocean or inland water body
        rather than on land.
    DUPLICATE_COORDINATE: Same coordinate is registered for
        multiple different plots.
    CLUSTER_ANOMALY: Multiple plots share the exact same
        coordinates, suggesting default or fabricated data.
    ELEVATION_IMPLAUSIBLE: Coordinate elevation exceeds plausible
        limits for the declared commodity type.
    ZERO_COORDINATE: Coordinate is at (0, 0) which is typically
        a default/null island value.
    HEMISPHERE_MISMATCH: Coordinate hemisphere does not match
        the expected hemisphere for the declared country.
    ROUNDED_COORDINATE: Coordinate appears to be rounded to whole
        degrees or single decimal place, indicating imprecision.
    """

    OUT_OF_BOUNDS = "out_of_bounds"
    LOW_PRECISION = "low_precision"
    LIKELY_TRANSPOSED = "likely_transposed"
    COUNTRY_MISMATCH = "country_mismatch"
    ON_WATER = "on_water"
    DUPLICATE_COORDINATE = "duplicate_coordinate"
    CLUSTER_ANOMALY = "cluster_anomaly"
    ELEVATION_IMPLAUSIBLE = "elevation_implausible"
    ZERO_COORDINATE = "zero_coordinate"
    HEMISPHERE_MISMATCH = "hemisphere_mismatch"
    ROUNDED_COORDINATE = "rounded_coordinate"


class PolygonIssueType(str, Enum):
    """Classification of polygon topology issues.

    Each issue type maps to a specific validation check performed
    by the Polygon Topology Verifier (Feature 2).

    UNCLOSED_RING: The polygon ring is not closed (first vertex
        does not match last vertex within tolerance).
    WRONG_WINDING_ORDER: Exterior ring is not counter-clockwise
        (CCW) or interior ring is not clockwise (CW).
    SELF_INTERSECTION: The polygon boundary crosses itself,
        creating an invalid geometry.
    INSUFFICIENT_VERTICES: Polygon has fewer than the minimum
        required vertices (default: 4, i.e. 3 unique + closure).
    EXCESSIVE_VERTICES: Polygon exceeds the maximum allowed
        vertex count (default: 100,000).
    AREA_MISMATCH: Calculated geodesic area differs from declared
        area by more than the tolerance percentage.
    SLIVER_POLYGON: Polygon has an extremely high perimeter-to-area
        ratio, indicating a thin sliver that is likely a data error.
    SPIKE_VERTEX: One or more vertices form extremely sharp angles
        (< 1 degree), indicating GPS noise or digitization errors.
    DUPLICATE_VERTEX: Consecutive vertices are at the same location
        (within floating-point tolerance).
    EXCEEDS_MAX_AREA: Polygon area exceeds the maximum plausible
        area for the declared commodity type.
    VERTEX_DENSITY_LOW: Average spacing between consecutive vertices
        is too large for accurate boundary representation.
    DEGENERATE_POLYGON: Polygon collapses to a point or line
        (area is effectively zero).
    """

    UNCLOSED_RING = "unclosed_ring"
    WRONG_WINDING_ORDER = "wrong_winding_order"
    SELF_INTERSECTION = "self_intersection"
    INSUFFICIENT_VERTICES = "insufficient_vertices"
    EXCESSIVE_VERTICES = "excessive_vertices"
    AREA_MISMATCH = "area_mismatch"
    SLIVER_POLYGON = "sliver_polygon"
    SPIKE_VERTEX = "spike_vertex"
    DUPLICATE_VERTEX = "duplicate_vertex"
    EXCEEDS_MAX_AREA = "exceeds_max_area"
    VERTEX_DENSITY_LOW = "vertex_density_low"
    DEGENERATE_POLYGON = "degenerate_polygon"


class OverlapSeverity(str, Enum):
    """Severity classification for protected area overlap.

    NONE: No overlap detected between the plot and any protected area.
    MARGINAL: Less than 10% of the plot area overlaps with a protected
        area. May be a boundary precision issue.
    PARTIAL: Between 10% and 90% of the plot area overlaps with a
        protected area. Requires investigation.
    FULL: More than 90% of the plot area overlaps with a protected
        area. Automatic compliance failure.
    """

    NONE = "none"
    MARGINAL = "marginal"
    PARTIAL = "partial"
    FULL = "full"


class DeforestationStatus(str, Enum):
    """Deforestation cutoff verification status for a plot.

    Classifies the forest status of a production plot relative to the
    EUDR cutoff date of December 31, 2020, per Article 2(1).

    VERIFIED_CLEAR: Plot was not forested at the cutoff date and has
        no deforestation concern. Agricultural land throughout the
        analysis period.
    VERIFIED_FOREST: Plot was forested at the cutoff date and remains
        forested. No deforestation has occurred, but commodity
        production from this plot may require additional verification.
    DEFORESTATION_DETECTED: Plot was forested at the cutoff date but
        tree cover has been lost after the cutoff. This is an automatic
        EUDR compliance failure.
    INCONCLUSIVE: Insufficient satellite data (cloud cover, data gaps)
        to make a definitive determination. Requires additional
        verification (e.g., field inspection).
    """

    VERIFIED_CLEAR = "verified_clear"
    VERIFIED_FOREST = "verified_forest"
    DEFORESTATION_DETECTED = "deforestation_detected"
    INCONCLUSIVE = "inconclusive"


class QualityTier(str, Enum):
    """Quality tier classification based on Geolocation Accuracy Score.

    Per PRD Appendix C, each tier has specific regulatory implications
    for Due Diligence Statement (DDS) eligibility.

    GOLD: Score 90-100. Fully verified, all checks passed.
        DDS-ready with minimal additional due diligence required.
    SILVER: Score 70-89. Mostly verified with minor issues.
        DDS-eligible with noted limitations.
    BRONZE: Score 50-69. Partially verified with significant issues.
        Enhanced due diligence required before DDS submission.
    UNVERIFIED: Score 0-49. Insufficient verification completed.
        Not eligible for DDS submission; remediation required.
    """

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    UNVERIFIED = "unverified"


class ChangeType(str, Enum):
    """Classification of temporal boundary change events.

    Tracks how a plot's boundary has changed between consecutive
    submissions, used by the Temporal Consistency Analyzer (Feature 6).

    EXPANSION: Plot area has increased by more than the configured
        threshold (default: 5%). May indicate encroachment.
    CONTRACTION: Plot area has decreased. Could indicate abandonment,
        sell-off, or data correction.
    SHIFT: Plot centroid has moved by more than the configured
        threshold (default: 100m) without significant area change.
        May indicate correction or data manipulation.
    RESHAPE: Plot area is similar but boundary shape has changed
        significantly. Vertices have been repositioned.
    """

    EXPANSION = "expansion"
    CONTRACTION = "contraction"
    SHIFT = "shift"
    RESHAPE = "reshape"


# =============================================================================
# Core Data Models
# =============================================================================


class CoordinateIssue(BaseModel):
    """A single issue detected during coordinate validation.

    Represents one specific problem found with a GPS coordinate pair,
    including the issue classification, severity, human-readable
    description, and any computed diagnostic values.

    Attributes:
        issue_type: Classification of the coordinate issue.
        severity: How critical this issue is (error or warning).
        description: Human-readable explanation of the issue.
        details: Additional diagnostic data (e.g., detected country,
            decimal places count, elevation value).
    """

    model_config = ConfigDict(from_attributes=True)

    issue_type: CoordinateIssueType = Field(
        ...,
        description="Classification of the coordinate issue",
    )
    severity: str = Field(
        default="error",
        description="Issue severity: 'error' or 'warning'",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the issue",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic data",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is error or warning."""
        v = v.lower().strip()
        if v not in ("error", "warning"):
            raise ValueError(
                f"severity must be 'error' or 'warning', got '{v}'"
            )
        return v


class PolygonIssue(BaseModel):
    """A single issue detected during polygon topology verification.

    Represents one specific problem found with a polygon geometry,
    including the issue classification, severity, affected vertex
    indices, and any computed diagnostic values.

    Attributes:
        issue_type: Classification of the polygon issue.
        severity: How critical this issue is (error or warning).
        description: Human-readable explanation of the issue.
        affected_vertices: Indices of vertices involved in the issue.
        details: Additional diagnostic data (e.g., intersection
            points, angle values, area calculations).
    """

    model_config = ConfigDict(from_attributes=True)

    issue_type: PolygonIssueType = Field(
        ...,
        description="Classification of the polygon issue",
    )
    severity: str = Field(
        default="error",
        description="Issue severity: 'error' or 'warning'",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the issue",
    )
    affected_vertices: List[int] = Field(
        default_factory=list,
        description="Indices of vertices involved in the issue",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic data",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is error or warning."""
        v = v.lower().strip()
        if v not in ("error", "warning"):
            raise ValueError(
                f"severity must be 'error' or 'warning', got '{v}'"
            )
        return v


class RepairSuggestion(BaseModel):
    """A suggested auto-repair action for a polygon issue.

    Provides actionable guidance for fixing common polygon topology
    issues, including the specific operation to perform and the
    expected outcome.

    Attributes:
        issue_type: The polygon issue type this repair addresses.
        repair_action: Short label for the repair (e.g., 'close_ring',
            'reverse_winding', 'remove_spike').
        description: Human-readable explanation of what the repair does.
        auto_repairable: Whether this repair can be applied
            automatically without human review.
        repaired_vertices: Suggested corrected vertex list if applicable.
    """

    model_config = ConfigDict(from_attributes=True)

    issue_type: PolygonIssueType = Field(
        ...,
        description="The polygon issue type this repair addresses",
    )
    repair_action: str = Field(
        ...,
        description="Short label for the repair action",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the repair",
    )
    auto_repairable: bool = Field(
        default=False,
        description="Whether this repair can be applied automatically",
    )
    repaired_vertices: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Suggested corrected vertex list if applicable",
    )


class ProtectedAreaOverlap(BaseModel):
    """Details of a detected overlap between a plot and a protected area.

    Contains identification, classification, and spatial metrics for
    a single protected area that overlaps with the verified plot.

    Attributes:
        protected_area_id: Unique identifier from the source database
            (e.g., WDPA ID).
        protected_area_name: Human-readable name of the protected area.
        protected_area_type: Type of protection (wdpa, ramsar, unesco,
            kba, icca, national).
        iucn_category: IUCN protected area category (Ia, Ib, II-VI)
            if applicable.
        designation_year: Year the area was officially designated.
        managing_authority: Name of the managing authority.
        overlap_percentage: Percentage of the plot area that overlaps
            with this protected area (0.0-100.0).
        overlap_area_hectares: Absolute overlap area in hectares.
        overlap_severity: Severity classification of the overlap.
    """

    model_config = ConfigDict(from_attributes=True)

    protected_area_id: str = Field(
        ...,
        description="Unique identifier from the source database",
    )
    protected_area_name: str = Field(
        ...,
        description="Human-readable name of the protected area",
    )
    protected_area_type: str = Field(
        ...,
        description="Type of protection: wdpa, ramsar, unesco, kba, icca, national",
    )
    iucn_category: Optional[str] = Field(
        None,
        description="IUCN protected area category (Ia, Ib, II-VI)",
    )
    designation_year: Optional[int] = Field(
        None,
        description="Year the area was officially designated",
    )
    managing_authority: Optional[str] = Field(
        None,
        description="Name of the managing authority",
    )
    overlap_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of plot area overlapping (0-100)",
    )
    overlap_area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Absolute overlap area in hectares",
    )
    overlap_severity: OverlapSeverity = Field(
        default=OverlapSeverity.NONE,
        description="Severity classification of the overlap",
    )


class ProtectedAreaProximity(BaseModel):
    """Details of a protected area found near (but not overlapping) a plot.

    Used for buffer zone analysis to identify plots that are close to
    protected areas even if they do not directly overlap.

    Attributes:
        protected_area_id: Unique identifier from the source database.
        protected_area_name: Human-readable name of the protected area.
        protected_area_type: Type of protection.
        iucn_category: IUCN category if applicable.
        distance_km: Distance in kilometers from the plot boundary
            to the nearest point of the protected area.
        direction: Cardinal direction from the plot to the protected
            area (N, NE, E, SE, S, SW, W, NW).
    """

    model_config = ConfigDict(from_attributes=True)

    protected_area_id: str = Field(
        ...,
        description="Unique identifier from the source database",
    )
    protected_area_name: str = Field(
        ...,
        description="Human-readable name of the protected area",
    )
    protected_area_type: str = Field(
        ...,
        description="Type of protection",
    )
    iucn_category: Optional[str] = Field(
        None,
        description="IUCN category if applicable",
    )
    distance_km: float = Field(
        ...,
        ge=0.0,
        description="Distance from plot to protected area in km",
    )
    direction: Optional[str] = Field(
        None,
        description="Cardinal direction from plot to protected area",
    )


class TreeCoverLossEvent(BaseModel):
    """A single tree cover loss event detected for a plot.

    Represents a discrete deforestation or tree cover loss event
    detected through satellite imagery analysis, including the
    data source, spatial extent, and confidence level.

    Attributes:
        event_date: Date when the tree cover loss was detected.
        tree_cover_loss_hectares: Area of tree cover loss in hectares.
        canopy_cover_before_pct: Canopy cover percentage before the event.
        canopy_cover_after_pct: Canopy cover percentage after the event.
        data_source: Satellite data source (e.g., 'hansen_gfc',
            'sentinel2', 'landsat', 'jaxa_palsar').
        confidence_score: Confidence in the detection (0-100).
        is_post_cutoff: Whether the event occurred after the EUDR
            cutoff date (December 31, 2020).
    """

    model_config = ConfigDict(from_attributes=True)

    event_date: Optional[date] = Field(
        None,
        description="Date when tree cover loss was detected",
    )
    tree_cover_loss_hectares: float = Field(
        ...,
        ge=0.0,
        description="Area of tree cover loss in hectares",
    )
    canopy_cover_before_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Canopy cover percentage before the event",
    )
    canopy_cover_after_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Canopy cover percentage after the event",
    )
    data_source: str = Field(
        ...,
        description="Satellite data source identifier",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence in the detection (0-100)",
    )
    is_post_cutoff: bool = Field(
        default=False,
        description="Whether event occurred after EUDR cutoff date",
    )


# =============================================================================
# Result Models
# =============================================================================


class CoordinateValidationResult(BaseModel):
    """Result of multi-layer coordinate validation (Feature 1).

    Contains the outcome of all coordinate validation checks including
    WGS84 bounds, precision assessment, country matching, ocean
    detection, duplicate detection, and elevation verification.

    Attributes:
        is_valid: Overall validation result (True if no errors).
        wgs84_bounds_ok: Whether coordinates are within valid WGS84 ranges.
        precision_score: Coordinate precision quality score (0-100).
        decimal_places: Number of decimal places in the coordinates.
        country_match: Whether coordinates fall within declared country.
        declared_country: ISO alpha-2 country code declared by user.
        detected_country: ISO alpha-2 country code detected from
            coordinates, or None if detection failed.
        is_on_land: Whether coordinates fall on land (not ocean).
        is_transposed: Whether latitude/longitude appear to be swapped.
        is_duplicate: Whether these coordinates duplicate another plot.
        elevation_m: Detected elevation in meters, if available.
        elevation_plausible: Whether elevation is plausible for the
            commodity type.
        issues: List of specific coordinate issues detected.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(
        ...,
        description="Overall validation result (True if no errors)",
    )
    wgs84_bounds_ok: bool = Field(
        default=True,
        description="Whether coordinates are within valid WGS84 ranges",
    )
    precision_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Coordinate precision quality score (0-100)",
    )
    decimal_places: int = Field(
        default=0,
        ge=0,
        description="Number of decimal places in the coordinates",
    )
    country_match: bool = Field(
        default=True,
        description="Whether coordinates fall within declared country",
    )
    declared_country: str = Field(
        default="",
        description="ISO alpha-2 country code declared by user",
    )
    detected_country: Optional[str] = Field(
        None,
        description="ISO alpha-2 country code detected from coordinates",
    )
    is_on_land: bool = Field(
        default=True,
        description="Whether coordinates fall on land",
    )
    is_transposed: bool = Field(
        default=False,
        description="Whether lat/lon appear to be swapped",
    )
    is_duplicate: bool = Field(
        default=False,
        description="Whether these coordinates duplicate another plot",
    )
    elevation_m: Optional[float] = Field(
        None,
        description="Detected elevation in meters",
    )
    elevation_plausible: bool = Field(
        default=True,
        description="Whether elevation is plausible for commodity type",
    )
    issues: List[CoordinateIssue] = Field(
        default_factory=list,
        description="List of specific coordinate issues detected",
    )


class PolygonVerificationResult(BaseModel):
    """Result of polygon topology verification (Feature 2).

    Contains the outcome of all polygon topology checks including
    ring closure, winding order, self-intersection, vertex count,
    area calculation, sliver detection, and spike detection.

    Attributes:
        is_valid: Overall topology validation result (True if no errors).
        is_closed: Whether the polygon ring is properly closed.
        winding_order_correct: Whether winding order follows the CCW
            exterior / CW holes convention.
        has_self_intersection: Whether the polygon self-intersects.
        vertex_count: Number of vertices in the polygon.
        min_vertex_count_met: Whether minimum vertex count is met.
        calculated_area_hectares: Geodesic area calculated using
            Karney's algorithm, in hectares.
        declared_area_hectares: Area declared by the data submitter.
        area_within_tolerance: Whether calculated area is within the
            configured tolerance of declared area.
        is_sliver: Whether polygon is classified as a sliver polygon.
        has_spike_vertices: Whether polygon has spike vertices.
        spike_vertex_indices: Indices of spike vertices detected.
        vertex_density_ok: Whether vertex spacing meets minimum
            requirements.
        issues: List of specific polygon issues detected.
        repair_suggestions: List of auto-repair suggestions.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(
        ...,
        description="Overall topology validation result",
    )
    is_closed: bool = Field(
        default=True,
        description="Whether the polygon ring is properly closed",
    )
    winding_order_correct: bool = Field(
        default=True,
        description="Whether winding order is correct (CCW exterior)",
    )
    has_self_intersection: bool = Field(
        default=False,
        description="Whether the polygon self-intersects",
    )
    vertex_count: int = Field(
        default=0,
        ge=0,
        description="Number of vertices in the polygon",
    )
    min_vertex_count_met: bool = Field(
        default=True,
        description="Whether minimum vertex count is met",
    )
    calculated_area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Calculated geodesic area in hectares",
    )
    declared_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Area declared by the data submitter",
    )
    area_within_tolerance: bool = Field(
        default=True,
        description="Whether area is within tolerance of declared",
    )
    is_sliver: bool = Field(
        default=False,
        description="Whether polygon is a sliver",
    )
    has_spike_vertices: bool = Field(
        default=False,
        description="Whether polygon has spike vertices",
    )
    spike_vertex_indices: List[int] = Field(
        default_factory=list,
        description="Indices of spike vertices detected",
    )
    vertex_density_ok: bool = Field(
        default=True,
        description="Whether vertex spacing meets minimum requirements",
    )
    issues: List[PolygonIssue] = Field(
        default_factory=list,
        description="List of specific polygon issues detected",
    )
    repair_suggestions: List[RepairSuggestion] = Field(
        default_factory=list,
        description="List of auto-repair suggestions",
    )


class ProtectedAreaCheckResult(BaseModel):
    """Result of protected area intersection analysis (Feature 3).

    Contains the outcome of cross-referencing the plot against all
    known protected area databases (WDPA, Ramsar, UNESCO, KBA, ICCA,
    national registries).

    Attributes:
        has_overlap: Whether any overlap was detected.
        overlapping_areas: List of overlapping protected areas with
            details and spatial metrics.
        buffer_zone_areas: List of protected areas within the buffer
            zone but not overlapping.
        total_overlap_percentage: Combined overlap percentage across
            all overlapping protected areas.
        overlap_severity: Highest severity classification among all
            overlaps detected.
        highest_protection_level: IUCN category of the most strictly
            protected overlapping area, if any.
    """

    model_config = ConfigDict(from_attributes=True)

    has_overlap: bool = Field(
        default=False,
        description="Whether any overlap was detected",
    )
    overlapping_areas: List[ProtectedAreaOverlap] = Field(
        default_factory=list,
        description="List of overlapping protected areas",
    )
    buffer_zone_areas: List[ProtectedAreaProximity] = Field(
        default_factory=list,
        description="Protected areas within buffer zone",
    )
    total_overlap_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Combined overlap percentage",
    )
    overlap_severity: OverlapSeverity = Field(
        default=OverlapSeverity.NONE,
        description="Highest overlap severity classification",
    )
    highest_protection_level: Optional[str] = Field(
        None,
        description="IUCN category of most strictly protected area",
    )


class DeforestationVerificationResult(BaseModel):
    """Result of deforestation cutoff verification (Feature 4).

    Contains the outcome of multi-temporal satellite analysis to verify
    that the plot was not converted from forest after December 31, 2020.

    Attributes:
        status: Deforestation status classification.
        canopy_cover_at_cutoff_pct: Canopy cover percentage at the
            EUDR cutoff date (December 31, 2020).
        canopy_cover_current_pct: Current canopy cover percentage.
        tree_cover_loss_events: List of discrete tree cover loss events
            detected across the analysis period.
        ndvi_baseline_2020: NDVI value at the cutoff date baseline.
        ndvi_current: Most recent NDVI value.
        data_sources_used: List of satellite data sources consulted.
        confidence_score: Confidence in the deforestation determination
            based on number of corroborating sources (0-100).
        evidence_package: Structured evidence data including before/after
            imagery dates, NDVI time series, and source metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    status: DeforestationStatus = Field(
        default=DeforestationStatus.INCONCLUSIVE,
        description="Deforestation status classification",
    )
    canopy_cover_at_cutoff_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Canopy cover at EUDR cutoff date (%)",
    )
    canopy_cover_current_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Current canopy cover percentage",
    )
    tree_cover_loss_events: List[TreeCoverLossEvent] = Field(
        default_factory=list,
        description="Discrete tree cover loss events",
    )
    ndvi_baseline_2020: Optional[float] = Field(
        None,
        description="NDVI value at cutoff date baseline",
    )
    ndvi_current: Optional[float] = Field(
        None,
        description="Most recent NDVI value",
    )
    data_sources_used: List[str] = Field(
        default_factory=list,
        description="Satellite data sources consulted",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence in deforestation determination (0-100)",
    )
    evidence_package: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured evidence data for audit",
    )


class GeolocationAccuracyScore(BaseModel):
    """Composite Geolocation Accuracy Score (GAS) for a single plot (Feature 5).

    Provides a weighted composite score from 0-100 based on six
    verification dimensions: coordinate precision, polygon quality,
    country match, protected area clearance, deforestation clearance,
    and temporal consistency.

    Attributes:
        plot_id: Unique identifier of the verified plot.
        total_score: Composite accuracy score (0-100).
        quality_tier: Quality tier classification based on total_score.
        coordinate_precision_score: Score for coordinate precision (0-20).
        polygon_quality_score: Score for polygon topology quality (0-20).
        country_match_score: Score for country boundary match (0-15).
        protected_area_score: Score for protected area clearance (0-15).
        deforestation_score: Score for deforestation clearance (0-15).
        temporal_consistency_score: Score for temporal consistency (0-15).
        component_details: Breakdown of individual check results.
        calculated_at: UTC timestamp of score calculation.
        provenance_hash: SHA-256 hash of the score calculation inputs
            and outputs for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        description="Unique identifier of the verified plot",
    )
    total_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite accuracy score (0-100)",
    )
    quality_tier: QualityTier = Field(
        default=QualityTier.UNVERIFIED,
        description="Quality tier classification",
    )
    coordinate_precision_score: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Coordinate precision score (0-20)",
    )
    polygon_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Polygon quality score (0-20)",
    )
    country_match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="Country match score (0-15)",
    )
    protected_area_score: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="Protected area clearance score (0-15)",
    )
    deforestation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="Deforestation clearance score (0-15)",
    )
    temporal_consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=15.0,
        description="Temporal consistency score (0-15)",
    )
    component_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Breakdown of individual check results",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of score calculation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )


class BoundaryChange(BaseModel):
    """A single boundary change event detected between plot versions.

    Describes how a plot's boundary changed between two consecutive
    submissions, including area change, centroid shift, expansion
    direction, and forest encroachment assessment.

    Attributes:
        change_type: Classification of the boundary change.
        area_before_hectares: Plot area before the change (hectares).
        area_after_hectares: Plot area after the change (hectares).
        area_change_pct: Percentage change in area.
        centroid_shift_meters: Distance the centroid moved (meters).
        expansion_direction: Cardinal direction of expansion, if any.
        expands_into_forest: Whether expansion encroaches into an area
            that was forested at the EUDR cutoff date.
        detected_at: UTC timestamp when the change was detected.
    """

    model_config = ConfigDict(from_attributes=True)

    change_type: ChangeType = Field(
        ...,
        description="Classification of the boundary change",
    )
    area_before_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Plot area before the change (hectares)",
    )
    area_after_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Plot area after the change (hectares)",
    )
    area_change_pct: float = Field(
        default=0.0,
        description="Percentage change in area",
    )
    centroid_shift_meters: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance the centroid moved (meters)",
    )
    expansion_direction: Optional[str] = Field(
        None,
        description="Cardinal direction of expansion (N/NE/E/SE/S/SW/W/NW)",
    )
    expands_into_forest: bool = Field(
        default=False,
        description="Whether expansion encroaches into forest",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when change was detected",
    )


class TemporalChangeResult(BaseModel):
    """Result of temporal consistency analysis for a plot (Feature 6).

    Contains the history of boundary changes, anomaly flags, and
    overall temporal consistency assessment.

    Attributes:
        plot_id: Unique identifier of the verified plot.
        total_versions: Number of boundary versions analyzed.
        boundary_changes: List of detected boundary change events.
        has_suspicious_changes: Whether any changes are flagged as
            potentially suspicious.
        rapid_changes_detected: Whether multiple changes occurred
            within a short time window (< 30 days).
        forest_encroachment_detected: Whether any boundary expansion
            encroaches into areas forested at cutoff.
        temporal_consistency_score: Temporal consistency component
            score (0-100).
        analysis_period_start: Start of the analysis period.
        analysis_period_end: End of the analysis period.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        description="Unique identifier of the verified plot",
    )
    total_versions: int = Field(
        default=1,
        ge=1,
        description="Number of boundary versions analyzed",
    )
    boundary_changes: List[BoundaryChange] = Field(
        default_factory=list,
        description="Detected boundary change events",
    )
    has_suspicious_changes: bool = Field(
        default=False,
        description="Whether any changes are flagged as suspicious",
    )
    rapid_changes_detected: bool = Field(
        default=False,
        description="Whether multiple changes occurred within 30 days",
    )
    forest_encroachment_detected: bool = Field(
        default=False,
        description="Whether expansion encroaches into forest",
    )
    temporal_consistency_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Temporal consistency score (0-100)",
    )
    analysis_period_start: Optional[datetime] = Field(
        None,
        description="Start of the analysis period",
    )
    analysis_period_end: Optional[datetime] = Field(
        None,
        description="End of the analysis period",
    )


# =============================================================================
# Request Models
# =============================================================================


class VerifyCoordinateRequest(BaseModel):
    """Request body for validating a single coordinate pair.

    Attributes:
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        declared_country_code: ISO 3166-1 alpha-2 country code as
            declared by the data submitter.
        commodity: EUDR commodity for context-aware validation.
        existing_plot_ids: Optional list of existing plot IDs to check
            for duplicate coordinates.
    """

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(
        ...,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        description="WGS84 longitude in decimal degrees",
    )
    declared_country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for context-aware validation",
    )
    existing_plot_ids: List[str] = Field(
        default_factory=list,
        description="Existing plot IDs for duplicate checking",
    )

    @field_validator("declared_country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate and normalize country code to uppercase ISO alpha-2."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "declared_country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v


class VerifyPolygonRequest(BaseModel):
    """Request body for verifying a single polygon geometry.

    Attributes:
        vertices: List of (latitude, longitude) tuples defining the
            polygon boundary. Must form a closed ring (first == last).
        declared_area_hectares: Area in hectares as declared by the
            data submitter. Used for tolerance checking.
        declared_country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity for context-aware area limits.
    """

    model_config = ConfigDict(extra="forbid")

    vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="Polygon vertices as (lat, lon) tuples",
    )
    declared_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Declared area in hectares",
    )
    declared_country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for context-aware validation",
    )

    @field_validator("declared_country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate and normalize country code to uppercase ISO alpha-2."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "declared_country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v


class VerifyPlotRequest(BaseModel):
    """Request body for full verification of a single production plot.

    Combines coordinate validation, polygon verification (if provided),
    and additional checks based on the verification level.

    Attributes:
        plot_id: Unique identifier for the plot being verified.
        coordinates: GPS (latitude, longitude) tuple in WGS84.
        polygon: Optional polygon vertices for plots > 4 hectares.
        declared_area_hectares: Declared area in hectares.
        declared_country_code: ISO 3166-1 alpha-2 country code.
        declared_region: Optional sub-national region.
        commodity: EUDR commodity produced on this plot.
        verification_level: Depth of verification to perform.
        operator_id: Optional operator ID for provenance tracking.
    """

    model_config = ConfigDict(extra="forbid")

    plot_id: str = Field(
        ...,
        description="Unique identifier for the plot",
    )
    coordinates: Tuple[float, float] = Field(
        ...,
        description="GPS coordinates as (latitude, longitude) in WGS84",
    )
    polygon: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Polygon vertices for plots > 4 hectares",
    )
    declared_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Declared area in hectares",
    )
    declared_country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    declared_region: Optional[str] = Field(
        None,
        max_length=200,
        description="Sub-national region or administrative division",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity produced on this plot",
    )
    verification_level: VerificationLevel = Field(
        default=VerificationLevel.STANDARD,
        description="Depth of verification to perform",
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

    @field_validator("declared_country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate and normalize country code to uppercase ISO alpha-2."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "declared_country_code must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Validate GPS coordinates are within valid WGS84 ranges."""
        lat, lon = v
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"Latitude must be between -90 and 90, got {lat}"
            )
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(
                f"Longitude must be between -180 and 180, got {lon}"
            )
        return v

    @model_validator(mode="after")
    def validate_polygon_for_large_plots(self) -> VerifyPlotRequest:
        """Check that plots > 4 hectares have polygon data (warning only).

        Per EUDR Article 9(1)(b-d), plots exceeding 4 hectares must
        provide polygon boundary data. This validator does not reject
        the request but the verification engine will flag this as an
        issue in the result.
        """
        return self


class BatchVerificationRequest(BaseModel):
    """Request body for submitting a batch verification job.

    Attributes:
        plots: List of individual plot verification requests.
        verification_level: Verification level to apply to all plots.
        priority_country_codes: Optional list of country codes to
            prioritize (verified first in the batch).
        modified_since: Optional datetime filter to only verify plots
            modified since this timestamp.
        operator_id: Operator ID for the batch job.
    """

    model_config = ConfigDict(extra="forbid")

    plots: List[VerifyPlotRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of plot verification requests",
    )
    verification_level: VerificationLevel = Field(
        default=VerificationLevel.STANDARD,
        description="Verification level for all plots",
    )
    priority_country_codes: List[str] = Field(
        default_factory=list,
        description="Country codes to prioritize",
    )
    modified_since: Optional[datetime] = Field(
        None,
        description="Only verify plots modified after this timestamp",
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


class ComplianceReportRequest(BaseModel):
    """Request body for generating an Article 9 compliance report.

    Attributes:
        operator_id: Operator ID for the compliance report.
        commodity: Optional commodity filter.
        country_code: Optional country filter.
        include_plot_details: Whether to include per-plot verification
            details in the report.
        report_format: Desired report format (json, csv, pdf).
    """

    model_config = ConfigDict(extra="forbid")

    operator_id: str = Field(
        ...,
        description="Operator ID for the compliance report",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="Optional commodity filter",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Optional country filter",
    )
    include_plot_details: bool = Field(
        default=True,
        description="Whether to include per-plot details",
    )
    report_format: str = Field(
        default="json",
        description="Report format: json, csv, or pdf",
    )

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

    @field_validator("report_format")
    @classmethod
    def validate_report_format(cls, v: str) -> str:
        """Validate report format is supported."""
        v = v.lower().strip()
        if v not in ("json", "csv", "pdf"):
            raise ValueError(
                f"report_format must be 'json', 'csv', or 'pdf', got '{v}'"
            )
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
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


# =============================================================================
# Response Models
# =============================================================================


class PlotVerificationResult(BaseModel):
    """Complete verification result for a single production plot.

    Aggregates results from all verification engines (coordinate,
    polygon, protected area, deforestation, temporal) into a single
    response with an overall status and accuracy score.

    Attributes:
        verification_id: Unique identifier for this verification run.
        plot_id: Identifier of the verified plot.
        operator_id: Operator who owns this plot.
        verification_level: Level of verification performed.
        overall_status: Overall verification outcome.
        accuracy_score: Composite Geolocation Accuracy Score.
        coordinate_result: Result of coordinate validation.
        polygon_result: Result of polygon topology verification.
        protected_area_result: Result of protected area screening.
        deforestation_result: Result of deforestation verification.
        temporal_result: Result of temporal consistency analysis.
        issues_count: Total number of issues detected across all checks.
        critical_issues_count: Number of critical (error) issues.
        provenance_hash: SHA-256 hash of the complete verification
            result for tamper detection.
        verified_at: UTC timestamp of verification completion.
        processing_time_ms: Total wall-clock processing time in ms.
    """

    model_config = ConfigDict(from_attributes=True)

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this verification run",
    )
    plot_id: str = Field(
        ...,
        description="Identifier of the verified plot",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator who owns this plot",
    )
    verification_level: VerificationLevel = Field(
        default=VerificationLevel.STANDARD,
        description="Level of verification performed",
    )
    overall_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Overall verification outcome",
    )
    accuracy_score: Optional[GeolocationAccuracyScore] = Field(
        None,
        description="Composite Geolocation Accuracy Score",
    )
    coordinate_result: Optional[CoordinateValidationResult] = Field(
        None,
        description="Result of coordinate validation",
    )
    polygon_result: Optional[PolygonVerificationResult] = Field(
        None,
        description="Result of polygon topology verification",
    )
    protected_area_result: Optional[ProtectedAreaCheckResult] = Field(
        None,
        description="Result of protected area screening",
    )
    deforestation_result: Optional[DeforestationVerificationResult] = Field(
        None,
        description="Result of deforestation verification",
    )
    temporal_result: Optional[TemporalChangeResult] = Field(
        None,
        description="Result of temporal consistency analysis",
    )
    issues_count: int = Field(
        default=0,
        ge=0,
        description="Total issues detected across all checks",
    )
    critical_issues_count: int = Field(
        default=0,
        ge=0,
        description="Number of critical (error) issues",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for tamper detection",
    )
    verified_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of verification completion",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )


class BatchProgress(BaseModel):
    """Real-time progress snapshot for a running batch verification job.

    Provides current processing status for UI integration via
    WebSocket or Server-Sent Events (SSE).

    Attributes:
        batch_id: Unique identifier of the batch job.
        total_plots: Total number of plots in the batch.
        processed: Number of plots processed so far.
        passed: Number of plots that passed verification.
        failed: Number of plots that failed verification.
        warnings: Number of plots that passed with warnings.
        pending: Number of plots still pending verification.
        progress_pct: Completion percentage (0-100).
        estimated_remaining_seconds: Estimated time to completion.
        current_plot_id: ID of the plot currently being processed.
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
    passed: int = Field(
        default=0,
        ge=0,
        description="Plots that passed verification",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Plots that failed verification",
    )
    warnings: int = Field(
        default=0,
        ge=0,
        description="Plots that passed with warnings",
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
        description="ID of the plot currently being processed",
    )


class BatchVerificationResult(BaseModel):
    """Complete result of a batch verification job.

    Provides aggregate statistics and individual plot results for
    a completed or partially completed batch verification.

    Attributes:
        batch_id: Unique identifier of the batch job.
        operator_id: Operator who submitted the batch.
        total_plots: Total number of plots in the batch.
        processed: Number of plots processed.
        passed: Number of plots that passed verification.
        failed: Number of plots that failed verification.
        warnings: Number of plots with warnings.
        pending: Number of plots still pending.
        verification_level: Verification level used for all plots.
        average_accuracy_score: Mean accuracy score across processed plots.
        results: List of individual plot verification results.
        started_at: UTC timestamp when the batch job started.
        completed_at: UTC timestamp when the batch job completed.
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
    passed: int = Field(
        default=0,
        ge=0,
        description="Plots that passed",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Plots that failed",
    )
    warnings: int = Field(
        default=0,
        ge=0,
        description="Plots with warnings",
    )
    pending: int = Field(
        default=0,
        ge=0,
        description="Plots still pending",
    )
    verification_level: VerificationLevel = Field(
        default=VerificationLevel.STANDARD,
        description="Verification level used",
    )
    average_accuracy_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Mean accuracy score across processed plots",
    )
    results: List[PlotVerificationResult] = Field(
        default_factory=list,
        description="Individual plot verification results",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
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


class ComplianceSummary(BaseModel):
    """Aggregate compliance statistics for a subset of plots.

    Used in compliance reports to summarize verification status by
    commodity, country, or operator.

    Attributes:
        category: Category label (commodity name, country code, etc.).
        total_plots: Total plots in this category.
        compliant_plots: Plots that passed verification.
        non_compliant_plots: Plots that failed verification.
        pending_plots: Plots not yet verified.
        compliance_rate: Percentage of plots that are compliant (0-100).
        average_accuracy_score: Mean accuracy score for this category.
        top_issues: Most common issue types in this category.
    """

    model_config = ConfigDict(from_attributes=True)

    category: str = Field(
        ...,
        description="Category label (commodity, country, etc.)",
    )
    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots in this category",
    )
    compliant_plots: int = Field(
        default=0,
        ge=0,
        description="Plots that passed verification",
    )
    non_compliant_plots: int = Field(
        default=0,
        ge=0,
        description="Plots that failed verification",
    )
    pending_plots: int = Field(
        default=0,
        ge=0,
        description="Plots not yet verified",
    )
    compliance_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Compliance rate percentage (0-100)",
    )
    average_accuracy_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Mean accuracy score for this category",
    )
    top_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most common issue types in this category",
    )


class ComplianceReport(BaseModel):
    """Complete Article 9 compliance report (Feature 8).

    Comprehensive report showing verification status, identified
    issues, remediation requirements, and compliance readiness scores
    per plot, per commodity, and per operator.

    Attributes:
        report_id: Unique identifier for this compliance report.
        operator_id: Operator for whom the report was generated.
        total_plots: Total plots covered by the report.
        compliant_plots: Number of compliant plots.
        non_compliant_plots: Number of non-compliant plots.
        pending_plots: Number of pending plots.
        overall_compliance_rate: Overall compliance percentage (0-100).
        average_accuracy_score: Mean accuracy score across all plots.
        commodity_summaries: Compliance breakdown by commodity.
        country_summaries: Compliance breakdown by country.
        top_remediation_priorities: Plots closest to compliance that
            should be remediated first.
        plot_results: Per-plot verification results (if requested).
        generated_at: UTC timestamp of report generation.
        report_format: Format of the report (json, csv, pdf).
        provenance_hash: SHA-256 hash of the complete report.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this compliance report",
    )
    operator_id: str = Field(
        ...,
        description="Operator for whom the report was generated",
    )
    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots covered by the report",
    )
    compliant_plots: int = Field(
        default=0,
        ge=0,
        description="Number of compliant plots",
    )
    non_compliant_plots: int = Field(
        default=0,
        ge=0,
        description="Number of non-compliant plots",
    )
    pending_plots: int = Field(
        default=0,
        ge=0,
        description="Number of pending plots",
    )
    overall_compliance_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall compliance percentage (0-100)",
    )
    average_accuracy_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Mean accuracy score across all plots",
    )
    commodity_summaries: List[ComplianceSummary] = Field(
        default_factory=list,
        description="Compliance breakdown by commodity",
    )
    country_summaries: List[ComplianceSummary] = Field(
        default_factory=list,
        description="Compliance breakdown by country",
    )
    top_remediation_priorities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Plots closest to compliance for prioritization",
    )
    plot_results: List[PlotVerificationResult] = Field(
        default_factory=list,
        description="Per-plot verification results",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of report generation",
    )
    report_format: str = Field(
        default="json",
        description="Format of the report (json, csv, pdf)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the complete report",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "VERSION",
    "EUDR_DEFORESTATION_CUTOFF",
    "DEFAULT_SCORE_WEIGHTS",
    "QUALITY_TIER_THRESHOLDS",
    "MAX_BATCH_SIZE",
    # Re-export for convenience
    "EUDRCommodity",
    # Enumerations
    "VerificationLevel",
    "VerificationStatus",
    "CoordinateIssueType",
    "PolygonIssueType",
    "OverlapSeverity",
    "DeforestationStatus",
    "QualityTier",
    "ChangeType",
    # Core models
    "CoordinateIssue",
    "PolygonIssue",
    "RepairSuggestion",
    "ProtectedAreaOverlap",
    "ProtectedAreaProximity",
    "TreeCoverLossEvent",
    # Result models
    "CoordinateValidationResult",
    "PolygonVerificationResult",
    "ProtectedAreaCheckResult",
    "DeforestationVerificationResult",
    "GeolocationAccuracyScore",
    "TemporalChangeResult",
    "BoundaryChange",
    # Request models
    "VerifyCoordinateRequest",
    "VerifyPolygonRequest",
    "VerifyPlotRequest",
    "BatchVerificationRequest",
    "ComplianceReportRequest",
    # Response models
    "PlotVerificationResult",
    "BatchVerificationResult",
    "BatchProgress",
    "ComplianceReport",
    "ComplianceSummary",
]
