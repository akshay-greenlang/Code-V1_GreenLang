# -*- coding: utf-8 -*-
"""
Land Use Change Detector Data Models - AGENT-EUDR-005

Pydantic v2 data models for the Land Use Change Detector Agent covering
land use classification, transition detection, temporal trajectory
analysis, EUDR cutoff date verification, cropland expansion detection,
conversion risk assessment, urban encroachment analysis, and compliance
reporting for EU Deforestation Regulation (EUDR) Articles 2, 9, 10,
and 12 compliance.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all land use change detection operations.

Enumerations (12):
    - LandUseCategory, TransitionType, TrajectoryType,
      ComplianceVerdict, ClassificationMethod, ConversionType,
      RiskTier, InfrastructureType, ReportType, ReportFormat,
      DataQualityLevel, BatchJobStatus

Core Models (10):
    - LandUseClassification, LandUseTransition, TransitionMatrix,
      TemporalTrajectory, CutoffVerification, CroplandConversion,
      ConversionRisk, UrbanEncroachment, ComplianceReport, BatchJob

Request Models (6):
    - ClassificationRequest, TransitionRequest, TrajectoryRequest,
      VerificationRequest, RiskAssessmentRequest, ReportRequest

Response Models (4):
    - ClassificationResponse, TransitionResponse,
      VerificationResponse, BatchResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, AGENT-EUDR-002
    Geolocation Verification, AGENT-EUDR-003 Satellite Monitoring,
    and AGENT-EUDR-004 Forest Cover Analysis.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

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
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Maximum number of parcels in a single batch analysis request.
MAX_BATCH_SIZE: int = 5000

#: Number of land use classes in the default classification scheme.
DEFAULT_NUM_CLASSES: int = 10

#: Minimum confidence score for classification acceptance.
MIN_CONFIDENCE_THRESHOLD: float = 0.60

#: Minimum transition area threshold in hectares.
MIN_TRANSITION_AREA_HA: float = 0.1

#: Maximum time steps supported in trajectory analysis.
MAX_TIME_STEPS: int = 60


# =============================================================================
# Enumerations
# =============================================================================


class LandUseCategory(str, Enum):
    """Land use category classification following IPCC and EUDR categories.

    Classifies land cover into 10 distinct categories for EUDR
    compliance assessment. Based on IPCC 2006 Guidelines for National
    Greenhouse Gas Inventories Volume 4 (AFOLU) land use categories
    with extensions for EUDR-specific analysis.

    FOREST: Land spanning more than 0.5 hectares with trees higher
        than 5 metres and canopy cover of more than 10 per cent per
        FAO definition. Includes primary, secondary, and plantation
        forest. Most relevant category for EUDR deforestation detection.
    SHRUBLAND: Land dominated by woody perennial plants (shrubs)
        typically less than 5 metres tall. Includes scrubland, bushland,
        and degraded forest transitioning to non-forest.
    GRASSLAND: Land covered with herbaceous plants with less than
        10 per cent tree and shrub cover. Includes natural grasslands,
        savanna, and improved pastures.
    CROPLAND: Land used for cultivation of crops, including annual
        and perennial crops, agroforestry with dominant crop component,
        and fallow land. Key conversion target for EUDR commodities.
    WETLAND: Land that is saturated or flooded with water for
        significant periods, including marshes, peatlands, and
        mangroves. Overlaps with forest for mangrove classification.
    WATER: Permanent water bodies including rivers, lakes, reservoirs,
        and coastal waters. Stable category rarely involved in
        transitions.
    URBAN: Built-up areas including residential, commercial,
        industrial, and transportation infrastructure. Relevant for
        urban encroachment analysis.
    BARE_SOIL: Land with minimal vegetation cover including rock
        outcrops, sandy deserts, and recently cleared land before
        vegetation establishment.
    SNOW_ICE: Permanent or semi-permanent snow and ice cover including
        glaciers and ice sheets. Relevant for high-latitude and
        high-altitude classifications.
    OTHER: Land that does not fit into the above categories, including
        mixed or transitional land use types requiring further
        investigation.
    """

    FOREST = "forest"
    SHRUBLAND = "shrubland"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    WETLAND = "wetland"
    WATER = "water"
    URBAN = "urban"
    BARE_SOIL = "bare_soil"
    SNOW_ICE = "snow_ice"
    OTHER = "other"


class TransitionType(str, Enum):
    """Land use transition type classification.

    Categorizes the type of land use change detected between two
    time periods. Each transition type has distinct spectral, temporal,
    and spatial characteristics that guide detection algorithms.

    DEFORESTATION: Conversion of forest to non-forest land use.
        The primary concern for EUDR compliance. Characterized by
        abrupt loss of canopy cover and change in spectral signature.
    DEGRADATION: Partial loss of forest cover or quality without
        complete conversion to non-forest. Includes selective logging,
        thinning, and fire damage.
    REFORESTATION: Conversion of non-forest to forest through active
        planting. Characterized by gradual increase in canopy cover
        over multiple years.
    NATURAL_REGROWTH: Spontaneous recovery of forest cover on
        previously cleared or degraded land without active planting.
        Slower than reforestation.
    CROPLAND_EXPANSION: Conversion of non-cropland (forest, grassland,
        shrubland) to agricultural cropland. Key EUDR transition for
        commodity-driven deforestation.
    URBAN_EXPANSION: Conversion of rural or agricultural land to
        urban built-up area. Characterized by increasing impervious
        surface fraction.
    WETLAND_CONVERSION: Drainage or filling of wetlands for
        agriculture or development. Includes mangrove clearing for
        aquaculture.
    STABLE: No significant land use change detected between time
        periods. Land use remains consistent within classification
        uncertainty.
    SEASONAL_CHANGE: Apparent land use change attributable to seasonal
        vegetation dynamics (e.g., deciduous leaf-off, crop rotation)
        rather than permanent conversion. Must be filtered from true
        transitions.
    UNKNOWN: Transition that cannot be classified with sufficient
        confidence. Requires additional data or manual review.
    """

    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    REFORESTATION = "reforestation"
    NATURAL_REGROWTH = "natural_regrowth"
    CROPLAND_EXPANSION = "cropland_expansion"
    URBAN_EXPANSION = "urban_expansion"
    WETLAND_CONVERSION = "wetland_conversion"
    STABLE = "stable"
    SEASONAL_CHANGE = "seasonal_change"
    UNKNOWN = "unknown"


class TrajectoryType(str, Enum):
    """Temporal trajectory type classification.

    Categorizes the pattern of land use or vegetation index change
    over time within a time series analysis.

    STABLE: Time series shows consistent values with no significant
        trend or breakpoints. Indicates persistent land use.
    ABRUPT_CHANGE: Time series shows a sudden, single-event change
        in values. Indicates rapid land use conversion such as
        clear-cutting or fire.
    GRADUAL_CHANGE: Time series shows a slow, continuous trend in
        values over multiple time steps. Indicates progressive
        degradation or gradual conversion.
    OSCILLATING: Time series shows repeated cyclical patterns
        without a clear directional trend. May indicate crop
        rotation cycles or seasonal variability.
    RECOVERY: Time series shows a significant drop followed by
        increasing values. Indicates vegetation regrowth or
        land restoration following a disturbance event.
    """

    STABLE = "stable"
    ABRUPT_CHANGE = "abrupt_change"
    GRADUAL_CHANGE = "gradual_change"
    OSCILLATING = "oscillating"
    RECOVERY = "recovery"


class ComplianceVerdict(str, Enum):
    """EUDR compliance verification verdict for land use change.

    The final regulatory determination for a parcel under EUDR
    Article 2. This verdict drives the DDS (Due Diligence Statement)
    outcome and determines whether commodities sourced from the
    parcel may be placed on the EU market.

    COMPLIANT: No prohibited land use change detected since the
        EUDR cutoff date (31 Dec 2020). Commodity may be placed
        on the EU market.
    NON_COMPLIANT: Prohibited land use change (deforestation or
        forest degradation) detected since the cutoff date.
        Commodity from this parcel must NOT be placed on the EU
        market.
    UNDER_REVIEW: Initial analysis indicates potential land use
        change that requires additional investigation or manual
        review before a final determination.
    INCONCLUSIVE: Insufficient data quality or confidence to make
        a definitive determination. Additional data collection
        required before DDS submission.
    EXEMPT: Parcel is exempt from EUDR requirements (e.g., land
        was already non-forest at the cutoff date and no forest
        existed within the regulatory lookback period).
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    INCONCLUSIVE = "inconclusive"
    EXEMPT = "exempt"


class ClassificationMethod(str, Enum):
    """Method for classifying land use from remote sensing data.

    Each method has trade-offs in accuracy, computational cost, and
    applicability to different biomes and data sources.

    SPECTRAL: Classification based on spectral reflectance patterns
        in multi-spectral satellite imagery. Uses band ratios and
        indices specific to each land use type.
    VEGETATION_INDEX: Classification using vegetation indices
        (NDVI, EVI, SAVI) with threshold-based or regression-based
        approaches for distinguishing vegetated from non-vegetated
        land use categories.
    PHENOLOGY: Classification using seasonal vegetation dynamics
        from multi-temporal imagery. Distinguishes crop types,
        deciduous forest, and other seasonal land uses.
    TEXTURE: Classification using spatial texture features derived
        from grey-level co-occurrence matrices (GLCM). Distinguishes
        structurally different land use types.
    ENSEMBLE: Weighted combination of multiple classification
        methods for robust, consensus-based classification with
        confidence estimates.
    """

    SPECTRAL = "spectral"
    VEGETATION_INDEX = "vegetation_index"
    PHENOLOGY = "phenology"
    TEXTURE = "texture"
    ENSEMBLE = "ensemble"


class ConversionType(str, Enum):
    """EUDR commodity-driven land use conversion type.

    Categorizes land use conversions by the EUDR-regulated commodity
    that is the primary driver of the conversion.

    CATTLE_RANCHING: Conversion of forest or other natural vegetation
        to cattle pasture. Common in South America (Brazil, Paraguay).
    COCOA_FARMING: Conversion of forest to cocoa plantation or
        cocoa agroforestry. Common in West Africa (Cote d'Ivoire,
        Ghana).
    COFFEE_CULTIVATION: Conversion of forest to coffee plantation.
        Common in Central America, East Africa, and Southeast Asia.
    PALM_OIL_PLANTATION: Conversion of forest to oil palm plantation.
        Common in Southeast Asia (Indonesia, Malaysia) and
        increasingly in West Africa and Central America.
    RUBBER_PLANTATION: Conversion of forest to rubber (Hevea
        brasiliensis) plantation. Common in Southeast Asia.
    SOYA_CULTIVATION: Conversion of forest or cerrado to soybean
        cultivation. Common in South America (Brazil, Argentina).
    TIMBER_HARVESTING: Conversion through commercial timber logging
        operations. May be selective or clear-cut.
    """

    CATTLE_RANCHING = "cattle_ranching"
    COCOA_FARMING = "cocoa_farming"
    COFFEE_CULTIVATION = "coffee_cultivation"
    PALM_OIL_PLANTATION = "palm_oil_plantation"
    RUBBER_PLANTATION = "rubber_plantation"
    SOYA_CULTIVATION = "soya_cultivation"
    TIMBER_HARVESTING = "timber_harvesting"


class RiskTier(str, Enum):
    """Risk tier classification for conversion risk assessment.

    Categorizes the level of deforestation or conversion risk for
    a parcel based on composite risk scoring across multiple factors
    including historical deforestation rate, commodity pressure,
    governance, and proximity to existing conversion fronts.

    LOW: Composite risk score 0-25%. Minimal risk of conversion.
        Low priority for monitoring and verification.
    MODERATE: Composite risk score 25-50%. Some risk factors present
        but overall risk is manageable. Standard monitoring frequency.
    HIGH: Composite risk score 50-75%. Significant risk factors
        present. Requires enhanced monitoring and verification.
    CRITICAL: Composite risk score 75-100%. Extreme risk of imminent
        or ongoing conversion. Requires immediate attention and
        highest-priority verification.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class InfrastructureType(str, Enum):
    """Infrastructure type classification for urban encroachment analysis.

    Categorizes the type of built infrastructure detected in
    proximity to forest or agricultural land.

    ROAD: Transportation roads including highways, secondary roads,
        and logging roads. Primary driver of forest access and
        subsequent deforestation.
    BUILDING: Residential, commercial, or industrial buildings.
        Indicates urbanization or settlement expansion.
    INDUSTRIAL: Industrial facilities including processing plants,
        warehouses, and mining operations.
    AGRICULTURAL_FACILITY: Agricultural infrastructure including
        silos, barns, irrigation systems, and processing facilities.
    TRANSPORTATION_HUB: Transportation hubs including ports, airports,
        rail terminals, and truck depots.
    """

    ROAD = "road"
    BUILDING = "building"
    INDUSTRIAL = "industrial"
    AGRICULTURAL_FACILITY = "agricultural_facility"
    TRANSPORTATION_HUB = "transportation_hub"


class ReportType(str, Enum):
    """Type of compliance report generated by the Land Use Change Detector.

    FULL: Complete compliance report with all analysis details,
        methodology description, evidence, and regulatory references.
    SUMMARY: Condensed summary report with key findings and
        verdict only.
    TRANSITION_ANALYSIS: Detailed report focusing on land use
        transition analysis with transition matrices and change maps.
    RISK_ASSESSMENT: Report focusing on conversion risk assessment
        with risk factor breakdowns and tier classification.
    REGULATORY_SUBMISSION: Report formatted for direct submission
        to EU regulatory authorities per EUDR implementing regulation.
    """

    FULL = "full"
    SUMMARY = "summary"
    TRANSITION_ANALYSIS = "transition_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    REGULATORY_SUBMISSION = "regulatory_submission"


class ReportFormat(str, Enum):
    """Output format for compliance reports.

    JSON: Machine-readable JSON format for API integration and
        downstream processing in GreenLang pipelines.
    PDF: Human-readable PDF report with maps, charts, and
        narrative summaries for regulatory submission.
    CSV: Tabular CSV format for bulk data export and spreadsheet
        analysis.
    EUDR_XML: EUDR-specific XML schema for direct submission to
        the EU Information System per Implementing Regulation.
    """

    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EUDR_XML = "eudr_xml"


class DataQualityLevel(str, Enum):
    """Data quality level classification for analysis confidence.

    Categorizes the overall data quality of an analysis based on
    composite scoring of temporal coverage, spatial resolution,
    source agreement, and cloud-free observation frequency.

    EXCELLENT: Overall quality score >90%. High confidence in all
        dimensions. Suitable for regulatory submission without
        caveats.
    GOOD: Overall quality score 70-90%. Good confidence with
        minor gaps. Suitable for regulatory submission with
        standard caveats.
    MODERATE: Overall quality score 50-70%. Moderate confidence
        with some limitations. May require supplementary evidence.
    POOR: Overall quality score 30-50%. Low confidence with
        significant gaps. Additional data collection recommended.
    INSUFFICIENT: Overall quality score <30%. Very low confidence.
        Not suitable for regulatory submission. Must collect
        additional data.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    INSUFFICIENT = "insufficient"


class BatchJobStatus(str, Enum):
    """Status of a batch land use change analysis job.

    Tracks the lifecycle of a batch job from creation through
    completion or failure.

    PENDING: Job has been created but not yet started processing.
        Queued for execution.
    RUNNING: Job is currently being processed. Some parcels may
        have completed while others are still in progress.
    COMPLETED: Job finished successfully with results for all
        parcels.
    FAILED: Job encountered a fatal error and could not complete.
        Partial results may be available.
    CANCELLED: Job was cancelled by the user or system before
        completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Core Data Models
# =============================================================================


class LandUseClassification(BaseModel):
    """Result of land use classification for a parcel.

    Contains the classified land use category, confidence score,
    classification method used, and per-class probability
    distribution from the classification operation.

    Attributes:
        parcel_id: Unique identifier for the classified parcel.
        primary_class: Primary land use category classification.
        primary_confidence: Confidence score for the primary
            classification (0.0-1.0).
        secondary_class: Secondary (runner-up) land use category.
        secondary_confidence: Confidence score for the secondary
            classification (0.0-1.0).
        class_probabilities: Dictionary mapping each land use
            category to its classification probability.
        method: Classification method used.
        classification_date: Date of the source imagery used for
            classification.
        spatial_resolution_m: Spatial resolution of the input
            imagery in metres.
        spectral_bands_used: List of spectral band identifiers
            used in classification.
        cloud_free_pct: Percentage of parcel area that was cloud-free
            in the source imagery.
        area_ha: Classified parcel area in hectares.
        centroid_lat: Latitude of the parcel centroid.
        centroid_lon: Longitude of the parcel centroid.
        provenance_hash: SHA-256 hash of the classification result.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Unique parcel identifier",
    )
    primary_class: LandUseCategory = Field(
        ...,
        description="Primary land use classification",
    )
    primary_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence for primary classification (0-1)",
    )
    secondary_class: Optional[LandUseCategory] = Field(
        None,
        description="Secondary land use classification",
    )
    secondary_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence for secondary classification (0-1)",
    )
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-class probability distribution",
    )
    method: ClassificationMethod = Field(
        default=ClassificationMethod.ENSEMBLE,
        description="Classification method used",
    )
    classification_date: Optional[date] = Field(
        None,
        description="Date of source imagery",
    )
    spatial_resolution_m: float = Field(
        default=10.0,
        gt=0.0,
        description="Input imagery spatial resolution (metres)",
    )
    spectral_bands_used: List[str] = Field(
        default_factory=list,
        description="Spectral bands used in classification",
    )
    cloud_free_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Cloud-free parcel area percentage",
    )
    area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Parcel area in hectares",
    )
    centroid_lat: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Parcel centroid latitude",
    )
    centroid_lon: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Parcel centroid longitude",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of classification result",
    )

    @model_validator(mode="after")
    def _validate_confidences(self) -> "LandUseClassification":
        """Validate primary confidence exceeds secondary confidence."""
        if (
            self.secondary_class is not None
            and self.secondary_confidence > self.primary_confidence
        ):
            raise ValueError(
                f"secondary_confidence ({self.secondary_confidence}) must "
                f"not exceed primary_confidence ({self.primary_confidence})"
            )
        return self


class LandUseTransition(BaseModel):
    """Detected land use transition between two time periods.

    Contains the from and to land use categories, transition type,
    detection confidence, area affected, and temporal details
    of the detected change.

    Attributes:
        transition_id: Unique identifier for this transition record.
        parcel_id: Parcel where the transition was detected.
        from_class: Land use category at the start of the period.
        to_class: Land use category at the end of the period.
        transition_type: Type of transition detected.
        confidence: Confidence in the transition detection (0-1).
        transition_area_ha: Area affected by the transition in ha.
        start_date: Start date of the observation period.
        end_date: End date of the observation period.
        estimated_transition_date: Best estimate of when the
            transition occurred within the observation period.
        magnitude: Magnitude of the change on a normalised scale
            (0.0-1.0), where 1.0 represents complete conversion.
        spectral_change_score: Composite spectral change metric
            quantifying the radiometric difference between periods.
        data_sources: List of data sources used for detection.
        provenance_hash: SHA-256 hash of the transition record.
    """

    model_config = ConfigDict(from_attributes=True)

    transition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transition identifier (UUID)",
    )
    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel where transition was detected",
    )
    from_class: LandUseCategory = Field(
        ...,
        description="Land use at start of period",
    )
    to_class: LandUseCategory = Field(
        ...,
        description="Land use at end of period",
    )
    transition_type: TransitionType = Field(
        ...,
        description="Type of transition detected",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence (0-1)",
    )
    transition_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Area affected by transition (hectares)",
    )
    start_date: Optional[date] = Field(
        None,
        description="Start of observation period",
    )
    end_date: Optional[date] = Field(
        None,
        description="End of observation period",
    )
    estimated_transition_date: Optional[date] = Field(
        None,
        description="Estimated date of transition",
    )
    magnitude: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised change magnitude (0-1)",
    )
    spectral_change_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Composite spectral change metric",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used for detection",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of transition record",
    )

    @model_validator(mode="after")
    def _validate_date_order(self) -> "LandUseTransition":
        """Validate temporal ordering of dates."""
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError(
                f"start_date ({self.start_date}) must be "
                f"<= end_date ({self.end_date})"
            )
        if (
            self.estimated_transition_date is not None
            and self.start_date is not None
            and self.estimated_transition_date < self.start_date
        ):
            raise ValueError(
                f"estimated_transition_date "
                f"({self.estimated_transition_date}) must be "
                f">= start_date ({self.start_date})"
            )
        if (
            self.estimated_transition_date is not None
            and self.end_date is not None
            and self.estimated_transition_date > self.end_date
        ):
            raise ValueError(
                f"estimated_transition_date "
                f"({self.estimated_transition_date}) must be "
                f"<= end_date ({self.end_date})"
            )
        return self


class TransitionMatrix(BaseModel):
    """Land use transition matrix for a region or set of parcels.

    A square matrix representing the area (in hectares) that
    transitioned between each pair of land use categories during
    the analysis period.

    Attributes:
        matrix_id: Unique identifier for this transition matrix.
        categories: Ordered list of land use categories (row/column
            labels) in the matrix.
        values: 2D matrix of transition areas in hectares.
            values[i][j] represents the area that changed from
            categories[i] to categories[j].
        total_area_ha: Total landscape area in hectares.
        total_change_area_ha: Total area that changed land use.
        start_date: Start date of the analysis period.
        end_date: End date of the analysis period.
        region_id: Optional identifier for the analysis region.
        parcel_count: Number of parcels included in the matrix.
        provenance_hash: SHA-256 hash of the transition matrix.
    """

    model_config = ConfigDict(from_attributes=True)

    matrix_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique matrix identifier (UUID)",
    )
    categories: List[str] = Field(
        ...,
        min_length=2,
        description="Ordered land use category labels",
    )
    values: List[List[float]] = Field(
        ...,
        description="2D transition area matrix (hectares)",
    )
    total_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Total landscape area (hectares)",
    )
    total_change_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Total area that changed land use (hectares)",
    )
    start_date: Optional[date] = Field(
        None,
        description="Start of analysis period",
    )
    end_date: Optional[date] = Field(
        None,
        description="End of analysis period",
    )
    region_id: str = Field(
        default="",
        description="Analysis region identifier",
    )
    parcel_count: int = Field(
        default=0,
        ge=0,
        description="Number of parcels in matrix",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of transition matrix",
    )

    @model_validator(mode="after")
    def _validate_matrix_dimensions(self) -> "TransitionMatrix":
        """Validate matrix is square and matches categories length."""
        n = len(self.categories)
        if len(self.values) != n:
            raise ValueError(
                f"Matrix row count ({len(self.values)}) must match "
                f"categories count ({n})"
            )
        for i, row in enumerate(self.values):
            if len(row) != n:
                raise ValueError(
                    f"Matrix row {i} has {len(row)} columns; "
                    f"expected {n}"
                )
        return self


class TemporalTrajectory(BaseModel):
    """Temporal trajectory analysis result for a parcel.

    Contains the classified trajectory type, time series values,
    breakpoints, and trend statistics from the temporal analysis
    of vegetation indices or land use signals.

    Attributes:
        parcel_id: Parcel analysed.
        trajectory_type: Classified trajectory pattern.
        time_steps: List of date strings representing each time
            step in the analysis.
        values: List of index values (e.g., NDVI) at each time step.
        breakpoints: List of indices into time_steps where
            significant changes were detected.
        trend_slope: Linear trend slope across the time series.
            Positive indicates increasing vegetation; negative
            indicates decreasing.
        trend_r_squared: Coefficient of determination for the
            linear trend fit.
        mean_value: Mean index value across the time series.
        std_dev: Standard deviation of index values.
        num_observations: Number of valid observations used.
        start_date: First date in the time series.
        end_date: Last date in the time series.
        index_name: Name of the vegetation or spectral index
            analysed (e.g., 'NDVI', 'EVI', 'NBR').
        confidence: Confidence in the trajectory classification.
        provenance_hash: SHA-256 hash of the trajectory result.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel identifier",
    )
    trajectory_type: TrajectoryType = Field(
        ...,
        description="Classified trajectory pattern",
    )
    time_steps: List[str] = Field(
        default_factory=list,
        description="Date strings for each time step",
    )
    values: List[float] = Field(
        default_factory=list,
        description="Index values at each time step",
    )
    breakpoints: List[int] = Field(
        default_factory=list,
        description="Indices of detected breakpoints",
    )
    trend_slope: float = Field(
        default=0.0,
        description="Linear trend slope",
    )
    trend_r_squared: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Trend R-squared (0-1)",
    )
    mean_value: float = Field(
        default=0.0,
        description="Mean index value",
    )
    std_dev: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of values",
    )
    num_observations: int = Field(
        default=0,
        ge=0,
        description="Number of valid observations",
    )
    start_date: Optional[date] = Field(
        None,
        description="First date in time series",
    )
    end_date: Optional[date] = Field(
        None,
        description="Last date in time series",
    )
    index_name: str = Field(
        default="NDVI",
        description="Vegetation/spectral index name",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Trajectory classification confidence (0-1)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of trajectory result",
    )

    @model_validator(mode="after")
    def _validate_time_series_length(self) -> "TemporalTrajectory":
        """Validate time_steps and values have matching length."""
        if self.time_steps and self.values:
            if len(self.time_steps) != len(self.values):
                raise ValueError(
                    f"time_steps length ({len(self.time_steps)}) must "
                    f"match values length ({len(self.values)})"
                )
        return self


class CutoffVerification(BaseModel):
    """EUDR cutoff date verification result for a parcel.

    Verifies the land use state of a parcel at the EUDR cutoff
    date (31 December 2020) by analysing imagery from the search
    window around the cutoff date.

    Attributes:
        parcel_id: Parcel verified.
        cutoff_date: EUDR cutoff date used for verification.
        land_use_at_cutoff: Classified land use at the cutoff date.
        land_use_current: Current classified land use.
        was_forest_at_cutoff: Whether the parcel was forest at
            the cutoff date per FAO definition.
        is_forest_current: Whether the parcel is currently forest.
        change_detected: Whether a land use change was detected
            between the cutoff date and current date.
        transition_type: Type of transition if change detected.
        imagery_date_used: Actual date of imagery closest to the
            cutoff date that was used for verification.
        days_from_cutoff: Number of days between the imagery date
            and the cutoff date (absolute value).
        confidence: Confidence in the cutoff verification (0-1).
        conservative_flag: True if conservative bias was applied
            (ambiguous cases classified as potential deforestation).
        evidence_sources: List of data sources used for verification.
        verdict: Compliance verdict based on the verification.
        provenance_hash: SHA-256 hash of the verification result.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel identifier",
    )
    cutoff_date: date = Field(
        default=EUDR_CUTOFF_DATE,
        description="EUDR cutoff date used",
    )
    land_use_at_cutoff: LandUseCategory = Field(
        ...,
        description="Land use at cutoff date",
    )
    land_use_current: LandUseCategory = Field(
        ...,
        description="Current land use",
    )
    was_forest_at_cutoff: bool = Field(
        ...,
        description="Was forest at cutoff (FAO definition)",
    )
    is_forest_current: bool = Field(
        ...,
        description="Currently forest (FAO definition)",
    )
    change_detected: bool = Field(
        ...,
        description="Land use change detected since cutoff",
    )
    transition_type: Optional[TransitionType] = Field(
        None,
        description="Transition type if change detected",
    )
    imagery_date_used: Optional[date] = Field(
        None,
        description="Actual imagery date closest to cutoff",
    )
    days_from_cutoff: int = Field(
        default=0,
        ge=0,
        description="Days between imagery and cutoff date",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Verification confidence (0-1)",
    )
    conservative_flag: bool = Field(
        default=False,
        description="Conservative bias was applied",
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Data sources for verification",
    )
    verdict: ComplianceVerdict = Field(
        ...,
        description="Compliance verdict from verification",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of verification result",
    )

    @model_validator(mode="after")
    def _validate_change_transition(self) -> "CutoffVerification":
        """Validate transition_type is set when change_detected is True."""
        if self.change_detected and self.transition_type is None:
            raise ValueError(
                "transition_type must be set when change_detected is True"
            )
        return self


class CroplandConversion(BaseModel):
    """Detected cropland expansion event for a parcel.

    Contains details of land use conversion to cropland, including
    the source land use, conversion area, associated EUDR commodity,
    and temporal information.

    Attributes:
        conversion_id: Unique identifier for this conversion event.
        parcel_id: Parcel where conversion was detected.
        from_class: Original land use category before conversion.
        conversion_type: EUDR commodity-driven conversion type.
        commodity: EUDR-regulated commodity associated with the
            conversion.
        conversion_area_ha: Area converted to cropland in hectares.
        conversion_date: Estimated date of conversion.
        detection_date: Date when the conversion was detected.
        confidence: Confidence in the conversion detection (0-1).
        is_post_cutoff: Whether the conversion occurred after the
            EUDR cutoff date (31 Dec 2020).
        previous_forest_pct: Percentage of the converted area that
            was previously forest.
        soil_suitability_score: Soil suitability score for the
            commodity crop (0-1). Higher indicates better suitability.
        proximity_to_existing_cropland_km: Distance to nearest
            existing cropland in kilometres.
        evidence_summary: Narrative summary of conversion evidence.
        provenance_hash: SHA-256 hash of the conversion record.
    """

    model_config = ConfigDict(from_attributes=True)

    conversion_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique conversion identifier (UUID)",
    )
    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel where conversion detected",
    )
    from_class: LandUseCategory = Field(
        ...,
        description="Original land use before conversion",
    )
    conversion_type: ConversionType = Field(
        ...,
        description="Commodity-driven conversion type",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="Associated EUDR commodity",
    )
    conversion_area_ha: float = Field(
        ...,
        ge=0.0,
        description="Converted area (hectares)",
    )
    conversion_date: Optional[date] = Field(
        None,
        description="Estimated conversion date",
    )
    detection_date: Optional[date] = Field(
        None,
        description="Date conversion was detected",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence (0-1)",
    )
    is_post_cutoff: bool = Field(
        default=False,
        description="Conversion after EUDR cutoff date",
    )
    previous_forest_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Previously forested percentage of converted area",
    )
    soil_suitability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Soil suitability for commodity (0-1)",
    )
    proximity_to_existing_cropland_km: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance to nearest cropland (km)",
    )
    evidence_summary: str = Field(
        default="",
        description="Narrative summary of conversion evidence",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of conversion record",
    )


class ConversionRisk(BaseModel):
    """Conversion risk assessment result for a parcel.

    Contains the composite risk score, risk tier classification,
    individual risk factor scores, and recommendations based on
    multi-factor risk modelling.

    Attributes:
        parcel_id: Parcel assessed.
        risk_score: Composite risk score (0.0-100.0).
        risk_tier: Classified risk tier.
        factor_scores: Dictionary mapping risk factor names to
            their individual scores (0.0-100.0).
        weighted_scores: Dictionary mapping risk factor names to
            their weighted contributions to the composite score.
        dominant_factor: Risk factor with the highest weighted
            contribution.
        commodity: EUDR commodity for commodity-specific risk
            assessment.
        historical_deforestation_rate: Annual deforestation rate
            in the area (percent per year).
        governance_index: Governance quality index for the region
            (0-1, higher is better governance).
        protected_area_overlap_pct: Percentage of parcel that
            overlaps with protected areas.
        assessment_date: Date of the risk assessment.
        recommendations: List of recommended actions based on
            risk level.
        provenance_hash: SHA-256 hash of the risk assessment.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel identifier",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite risk score (0-100)",
    )
    risk_tier: RiskTier = Field(
        ...,
        description="Classified risk tier",
    )
    factor_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual risk factor scores (0-100)",
    )
    weighted_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Weighted factor contributions",
    )
    dominant_factor: str = Field(
        default="",
        description="Highest-weighted risk factor",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for assessment",
    )
    historical_deforestation_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Annual deforestation rate (%/year)",
    )
    governance_index: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Governance quality index (0-1)",
    )
    protected_area_overlap_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Protected area overlap percentage",
    )
    assessment_date: Optional[date] = Field(
        None,
        description="Date of risk assessment",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions based on risk",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of risk assessment",
    )

    @field_validator("risk_tier", mode="before")
    @classmethod
    def _validate_risk_tier(cls, v: Any) -> Any:
        """Accept risk tier as string or enum."""
        return v

    @model_validator(mode="after")
    def _validate_tier_consistency(self) -> "ConversionRisk":
        """Validate risk tier is consistent with risk score."""
        expected = _score_to_risk_tier(self.risk_score)
        if self.risk_tier != expected:
            self.risk_tier = expected
        return self


def _score_to_risk_tier(score: float) -> RiskTier:
    """Map a risk score to a RiskTier.

    Args:
        score: Risk score (0-100).

    Returns:
        Corresponding RiskTier.
    """
    if score >= 75.0:
        return RiskTier.CRITICAL
    elif score >= 50.0:
        return RiskTier.HIGH
    elif score >= 25.0:
        return RiskTier.MODERATE
    else:
        return RiskTier.LOW


class UrbanEncroachment(BaseModel):
    """Urban encroachment analysis result for a parcel.

    Contains the analysis of urban or infrastructure expansion
    in proximity to forested or agricultural land, including
    detected infrastructure types, distances, and growth rates.

    Attributes:
        parcel_id: Parcel analysed.
        buffer_km: Buffer distance used for analysis (km).
        urban_area_within_buffer_ha: Total urban area within
            the buffer zone in hectares.
        urban_growth_rate_pct_per_year: Annual urban growth rate
            within the buffer zone (percent per year).
        nearest_urban_distance_km: Distance to the nearest urban
            area from the parcel centroid (km).
        infrastructure_detected: List of infrastructure types
            detected within the buffer zone.
        infrastructure_density: Dictionary mapping infrastructure
            types to their count within the buffer.
        road_density_km_per_sq_km: Road network density within
            the buffer (km of road per sq km of area).
        population_estimate: Estimated population within the
            buffer zone.
        population_growth_rate: Annual population growth rate
            within the buffer (percent per year).
        encroachment_risk_score: Composite encroachment risk
            score (0-100).
        analysis_date: Date of the encroachment analysis.
        data_sources: List of data sources used.
        provenance_hash: SHA-256 hash of the analysis result.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel identifier",
    )
    buffer_km: float = Field(
        ...,
        gt=0.0,
        description="Buffer distance used (km)",
    )
    urban_area_within_buffer_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Urban area within buffer (hectares)",
    )
    urban_growth_rate_pct_per_year: float = Field(
        default=0.0,
        description="Annual urban growth rate (%/year)",
    )
    nearest_urban_distance_km: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance to nearest urban area (km)",
    )
    infrastructure_detected: List[InfrastructureType] = Field(
        default_factory=list,
        description="Infrastructure types detected in buffer",
    )
    infrastructure_density: Dict[str, int] = Field(
        default_factory=dict,
        description="Infrastructure count by type",
    )
    road_density_km_per_sq_km: float = Field(
        default=0.0,
        ge=0.0,
        description="Road density (km/sq km)",
    )
    population_estimate: int = Field(
        default=0,
        ge=0,
        description="Estimated population in buffer",
    )
    population_growth_rate: float = Field(
        default=0.0,
        description="Annual population growth rate (%/year)",
    )
    encroachment_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Encroachment risk score (0-100)",
    )
    analysis_date: Optional[date] = Field(
        None,
        description="Date of encroachment analysis",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of analysis result",
    )


class ComplianceReport(BaseModel):
    """Generated compliance report for EUDR land use change assessment.

    Contains metadata for a compliance report generated from land
    use change detection results, including provenance tracking for
    regulatory audit trails.

    Attributes:
        report_id: Unique report identifier (UUID).
        report_type: Type of report generated.
        format: Output format of the report.
        parcel_id: Parcel identifier this report covers.
        verdict: EUDR compliance verdict for the parcel.
        summary: Brief narrative summary of report findings.
        transitions_detected: Number of land use transitions
            detected in the analysis period.
        risk_tier: Conversion risk tier for the parcel.
        created_at: UTC timestamp when the report was generated.
        provenance_hash: SHA-256 hash of the report content.
        regulatory_framework: Regulatory framework reference
            (default: EUDR EU 2023/1115).
        valid_until: Date until which this report is considered
            valid (typically 12 months from creation).
        reviewer: Optional reviewer identifier for approved reports.
        commodity: EUDR commodity associated with the parcel.
        data_quality: Data quality level of the analysis.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier (UUID)",
    )
    report_type: ReportType = Field(
        default=ReportType.FULL,
        description="Report type",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format",
    )
    parcel_id: str = Field(
        ...,
        description="Parcel this report covers",
    )
    verdict: ComplianceVerdict = Field(
        ...,
        description="EUDR compliance verdict",
    )
    summary: str = Field(
        default="",
        description="Brief narrative summary",
    )
    transitions_detected: int = Field(
        default=0,
        ge=0,
        description="Number of transitions detected",
    )
    risk_tier: Optional[RiskTier] = Field(
        None,
        description="Conversion risk tier",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of report generation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of report content",
    )
    regulatory_framework: str = Field(
        default="EUDR EU 2023/1115",
        description="Applicable regulatory framework",
    )
    valid_until: Optional[date] = Field(
        None,
        description="Validity expiration date",
    )
    reviewer: str = Field(
        default="",
        description="Reviewer identifier for approved reports",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="Associated EUDR commodity",
    )
    data_quality: Optional[DataQualityLevel] = Field(
        None,
        description="Data quality level of the analysis",
    )


class BatchJob(BaseModel):
    """Batch land use change analysis job.

    Tracks a batch processing job that analyses multiple parcels
    for land use change, transitions, and compliance verification.

    Attributes:
        job_id: Unique batch job identifier (UUID).
        status: Current job status.
        parcel_ids: List of parcel identifiers to analyse.
        total_parcels: Total number of parcels in the batch.
        completed_parcels: Number of parcels completed.
        failed_parcels: Number of parcels that failed.
        commodity: EUDR commodity for all parcels in the batch.
        analysis_types: Types of analysis to perform per parcel.
        priority: Processing priority (1=lowest, 10=highest).
        created_at: UTC timestamp when the job was created.
        started_at: UTC timestamp when processing started.
        completed_at: UTC timestamp when processing finished.
        error_messages: Dictionary mapping parcel_id to error.
        progress_pct: Completion percentage (0-100).
        estimated_remaining_seconds: Estimated time remaining.
        provenance_hash: SHA-256 hash of the batch job state.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier (UUID)",
    )
    status: BatchJobStatus = Field(
        default=BatchJobStatus.PENDING,
        description="Current job status",
    )
    parcel_ids: List[str] = Field(
        default_factory=list,
        description="Parcel identifiers to analyse",
    )
    total_parcels: int = Field(
        default=0,
        ge=0,
        description="Total parcels in batch",
    )
    completed_parcels: int = Field(
        default=0,
        ge=0,
        description="Parcels completed successfully",
    )
    failed_parcels: int = Field(
        default=0,
        ge=0,
        description="Parcels that failed analysis",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for batch",
    )
    analysis_types: List[str] = Field(
        default_factory=lambda: [
            "classification", "transition", "verification",
        ],
        description="Analysis types per parcel",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Processing priority (1-10)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Job creation timestamp (UTC)",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Processing start timestamp (UTC)",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Processing completion timestamp (UTC)",
    )
    error_messages: Dict[str, str] = Field(
        default_factory=dict,
        description="Errors keyed by parcel_id",
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
        description="Estimated time remaining (seconds)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of batch job state",
    )

    @field_validator("parcel_ids")
    @classmethod
    def _validate_batch_size(cls, v: List[str]) -> List[str]:
        """Validate batch size does not exceed maximum."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {MAX_BATCH_SIZE}"
            )
        return v


# =============================================================================
# Request Models
# =============================================================================


class ClassificationRequest(BaseModel):
    """Request to classify land use for a parcel.

    Attributes:
        parcel_id: Unique parcel identifier for tracking.
        polygon_wkt: Well-Known Text (WKT) representation of the
            parcel boundary polygon.
        imagery_date: Target date for classification. If None,
            uses the most recent available imagery.
        method: Classification method to use.
        include_probabilities: Whether to include full per-class
            probability distribution in the result.
        min_confidence: Minimum confidence threshold for accepting
            the classification. Overrides config default.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Unique parcel identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Parcel boundary polygon in WKT format",
    )
    imagery_date: Optional[date] = Field(
        None,
        description="Target date for classification",
    )
    method: ClassificationMethod = Field(
        default=ClassificationMethod.ENSEMBLE,
        description="Classification method",
    )
    include_probabilities: bool = Field(
        default=True,
        description="Include per-class probabilities",
    )
    min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence override",
    )


class TransitionRequest(BaseModel):
    """Request to detect land use transitions for a parcel.

    Attributes:
        parcel_id: Unique parcel identifier for tracking.
        polygon_wkt: WKT representation of the parcel boundary.
        start_date: Start date of the observation period.
        end_date: End date of the observation period.
        min_transition_area_ha: Minimum area for transitions to be
            reported. Overrides config default.
        include_matrix: Whether to include a transition matrix in
            the response.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Unique parcel identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Parcel boundary polygon in WKT format",
    )
    start_date: date = Field(
        ...,
        description="Start of observation period",
    )
    end_date: date = Field(
        ...,
        description="End of observation period",
    )
    min_transition_area_ha: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum transition area override (ha)",
    )
    include_matrix: bool = Field(
        default=False,
        description="Include transition matrix",
    )

    @model_validator(mode="after")
    def _validate_date_range(self) -> "TransitionRequest":
        """Validate start_date <= end_date."""
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be "
                f"<= end_date ({self.end_date})"
            )
        return self


class TrajectoryRequest(BaseModel):
    """Request to analyse temporal trajectory for a parcel.

    Attributes:
        parcel_id: Unique parcel identifier for tracking.
        polygon_wkt: WKT representation of the parcel boundary.
        start_date: Start date of the trajectory analysis.
        end_date: End date of the trajectory analysis.
        index_name: Vegetation or spectral index to analyse.
        max_time_steps: Maximum number of time steps. Overrides
            config default.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Unique parcel identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Parcel boundary in WKT format",
    )
    start_date: date = Field(
        ...,
        description="Start of trajectory analysis",
    )
    end_date: date = Field(
        ...,
        description="End of trajectory analysis",
    )
    index_name: str = Field(
        default="NDVI",
        description="Vegetation index to analyse",
    )
    max_time_steps: Optional[int] = Field(
        None,
        ge=2,
        le=1000,
        description="Maximum time steps override",
    )

    @model_validator(mode="after")
    def _validate_date_range(self) -> "TrajectoryRequest":
        """Validate start_date <= end_date."""
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be "
                f"<= end_date ({self.end_date})"
            )
        return self


class VerificationRequest(BaseModel):
    """Request to verify EUDR cutoff date compliance for a parcel.

    Attributes:
        parcel_id: Unique parcel identifier for tracking.
        polygon_wkt: WKT representation of the parcel boundary.
        commodity: EUDR-regulated commodity sourced from the parcel.
        use_conservative_bias: Whether to apply conservative bias
            for ambiguous cases. Overrides config default.
        include_evidence: Whether to include detailed evidence
            narrative in the result.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Unique parcel identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Parcel boundary in WKT format",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity from this parcel",
    )
    use_conservative_bias: Optional[bool] = Field(
        None,
        description="Conservative bias override",
    )
    include_evidence: bool = Field(
        default=True,
        description="Include evidence narrative",
    )


class RiskAssessmentRequest(BaseModel):
    """Request to assess conversion risk for a parcel.

    Attributes:
        parcel_id: Unique parcel identifier for tracking.
        polygon_wkt: WKT representation of the parcel boundary.
        commodity: EUDR-regulated commodity for risk scoping.
        include_recommendations: Whether to include action
            recommendations in the result.
        custom_weights: Optional custom risk factor weights to
            use instead of config defaults.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Unique parcel identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Parcel boundary in WKT format",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for risk scoping",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include action recommendations",
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Custom risk factor weights",
    )

    @field_validator("custom_weights")
    @classmethod
    def _validate_custom_weights(
        cls, v: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        """Validate custom weights sum to 1.0 if provided."""
        if v is not None:
            weight_sum = sum(v.values())
            if abs(weight_sum - 1.0) > 0.01:
                raise ValueError(
                    f"custom_weights must sum to 1.0, "
                    f"got {weight_sum:.4f}"
                )
            for wname, wval in v.items():
                if wval < 0.0 or wval > 1.0:
                    raise ValueError(
                        f"custom_weight '{wname}' must be in "
                        f"[0.0, 1.0], got {wval}"
                    )
        return v


class ReportRequest(BaseModel):
    """Request to generate a compliance report for a parcel.

    Attributes:
        parcel_id: Parcel identifier to generate report for.
        report_type: Type of report to generate.
        format: Output format for the report.
        include_maps: Whether to include map visualisations.
        commodity: EUDR commodity for the report.
    """

    model_config = ConfigDict(from_attributes=True)

    parcel_id: str = Field(
        ...,
        min_length=1,
        description="Parcel identifier",
    )
    report_type: ReportType = Field(
        default=ReportType.FULL,
        description="Report type",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format",
    )
    include_maps: bool = Field(
        default=False,
        description="Include map visualisations",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity for report",
    )


# =============================================================================
# Response Models
# =============================================================================


class ClassificationResponse(BaseModel):
    """Response from a land use classification request.

    Attributes:
        classification: The classification result.
        processing_time_ms: Processing time in milliseconds.
        data_quality: Data quality level of the classification.
        warnings: List of warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    classification: LandUseClassification = Field(
        ...,
        description="Classification result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    data_quality: Optional[DataQualityLevel] = Field(
        None,
        description="Data quality level",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class TransitionResponse(BaseModel):
    """Response from a land use transition detection request.

    Attributes:
        transitions: List of detected transitions.
        matrix: Optional transition matrix.
        total_change_area_ha: Total area that changed land use.
        processing_time_ms: Processing time in milliseconds.
        data_quality: Data quality level.
        warnings: List of warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    transitions: List[LandUseTransition] = Field(
        default_factory=list,
        description="Detected transitions",
    )
    matrix: Optional[TransitionMatrix] = Field(
        None,
        description="Transition matrix (if requested)",
    )
    total_change_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Total change area (hectares)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    data_quality: Optional[DataQualityLevel] = Field(
        None,
        description="Data quality level",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class VerificationResponse(BaseModel):
    """Response from a cutoff date verification request.

    Attributes:
        verification: The cutoff verification result.
        trajectory: Optional temporal trajectory analysis.
        risk: Optional conversion risk assessment.
        processing_time_ms: Processing time in milliseconds.
        data_quality: Data quality level.
        warnings: List of warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    verification: CutoffVerification = Field(
        ...,
        description="Cutoff verification result",
    )
    trajectory: Optional[TemporalTrajectory] = Field(
        None,
        description="Temporal trajectory (if available)",
    )
    risk: Optional[ConversionRisk] = Field(
        None,
        description="Conversion risk (if available)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    data_quality: Optional[DataQualityLevel] = Field(
        None,
        description="Data quality level",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class BatchResponse(BaseModel):
    """Response from a batch land use change analysis request.

    Attributes:
        job: The batch job tracking object.
        classifications: Completed classifications keyed by parcel_id.
        verifications: Completed verifications keyed by parcel_id.
        processing_time_ms: Total processing time in milliseconds.
        summary: Aggregate summary statistics.
    """

    model_config = ConfigDict(from_attributes=True)

    job: BatchJob = Field(
        ...,
        description="Batch job tracking object",
    )
    classifications: Dict[str, LandUseClassification] = Field(
        default_factory=dict,
        description="Classifications by parcel_id",
    )
    verifications: Dict[str, CutoffVerification] = Field(
        default_factory=dict,
        description="Verifications by parcel_id",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time (milliseconds)",
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate summary statistics",
    )


# ---------------------------------------------------------------------------
# Helper: DataQualityLevel mapping
# ---------------------------------------------------------------------------


def _score_to_quality_level(score: float) -> DataQualityLevel:
    """Map a quality score to a DataQualityLevel.

    Args:
        score: Quality score (0-100).

    Returns:
        Corresponding DataQualityLevel.
    """
    if score > 90.0:
        return DataQualityLevel.EXCELLENT
    elif score > 70.0:
        return DataQualityLevel.GOOD
    elif score > 50.0:
        return DataQualityLevel.MODERATE
    elif score > 30.0:
        return DataQualityLevel.POOR
    else:
        return DataQualityLevel.INSUFFICIENT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "DEFAULT_NUM_CLASSES",
    "MIN_CONFIDENCE_THRESHOLD",
    "MIN_TRANSITION_AREA_HA",
    "MAX_TIME_STEPS",
    # Re-exported
    "EUDRCommodity",
    # Enumerations
    "LandUseCategory",
    "TransitionType",
    "TrajectoryType",
    "ComplianceVerdict",
    "ClassificationMethod",
    "ConversionType",
    "RiskTier",
    "InfrastructureType",
    "ReportType",
    "ReportFormat",
    "DataQualityLevel",
    "BatchJobStatus",
    # Core models
    "LandUseClassification",
    "LandUseTransition",
    "TransitionMatrix",
    "TemporalTrajectory",
    "CutoffVerification",
    "CroplandConversion",
    "ConversionRisk",
    "UrbanEncroachment",
    "ComplianceReport",
    "BatchJob",
    # Request models
    "ClassificationRequest",
    "TransitionRequest",
    "TrajectoryRequest",
    "VerificationRequest",
    "RiskAssessmentRequest",
    "ReportRequest",
    # Response models
    "ClassificationResponse",
    "TransitionResponse",
    "VerificationResponse",
    "BatchResponse",
    # Helper functions
    "_score_to_risk_tier",
    "_score_to_quality_level",
]
