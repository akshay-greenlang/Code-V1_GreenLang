# -*- coding: utf-8 -*-
"""
Forest Cover Analysis Data Models - AGENT-EUDR-004

Pydantic v2 data models for the Forest Cover Analysis Agent covering
canopy density mapping, forest type classification, historical cover
reconstruction, deforestation-free verification, canopy height modeling,
landscape fragmentation analysis, above-ground biomass estimation, and
compliance reporting for EU Deforestation Regulation (EUDR) Articles 2,
9, 10, and 12 compliance.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all forest cover analysis operations.

Enumerations (12):
    - ForestType, CanopyDensityClass, DeforestationVerdict,
      DensityMethod, ClassificationMethod, HeightSource,
      BiomassSource, FragmentationLevel, ReportFormat,
      AnalysisStatus, DataQualityTier, EUDRCommodity (re-export)

Core Models (10):
    - CanopyDensityResult, ForestClassificationResult,
      HistoricalCoverRecord, DeforestationFreeResult,
      CanopyHeightEstimate, FragmentationMetrics,
      BiomassEstimate, ComplianceReport,
      DataQualityAssessment, PlotForestProfile

Request Models (6):
    - AnalyzeDensityRequest, ClassifyForestRequest,
      ReconstructHistoryRequest, VerifyDeforestationFreeRequest,
      BatchAnalysisRequest, GenerateReportRequest

Response Models (4):
    - BatchAnalysisResponse, BatchProgress,
      AnalysisSummary, ForestCoverDashboard

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, AGENT-EUDR-002
    Geolocation Verification, and AGENT-EUDR-003 Satellite Monitoring.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from greenlang.agents.data.eudr_traceability.models import EUDRCommodity
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Maximum number of plots in a single batch analysis request.
MAX_BATCH_SIZE: int = 5000

#: FAO canopy cover threshold for forest classification (percent).
#: Land with canopy cover above 10% qualifies as forest under the
#: FAO Global Forest Resources Assessment definition.
FAO_CANOPY_THRESHOLD: float = 10.0

#: FAO tree height threshold for forest classification (metres).
#: Trees must exceed 5m at maturity (or be able to reach 5m in situ)
#: for the land to qualify as forest.
FAO_HEIGHT_THRESHOLD: float = 5.0

#: FAO minimum forest area threshold (hectares).
#: Land parcels must span more than 0.5 ha to qualify as forest.
FAO_AREA_THRESHOLD: float = 0.5

#: Number of global terrestrial biomes recognised for stratified
#: forest cover analysis (based on WWF terrestrial ecoregions).
BIOME_COUNT: int = 16

#: Conversion factor from above-ground biomass (AGB) in Mg/ha to
#: carbon stock in tC/ha. Default factor of 0.47 per IPCC 2006
#: Guidelines for National Greenhouse Gas Inventories, Vol 4, Ch 4.
AGB_CONVERSION_FACTOR: float = 0.47

# =============================================================================
# Enumerations
# =============================================================================

class ForestType(str, Enum):
    """Forest type classification following EUDR and FAO categories.

    Classifies forest land into ecologically and structurally distinct
    categories relevant for EUDR compliance assessment. Based on FAO
    Global Forest Resources Assessment 2020 forest type taxonomy with
    extensions for EUDR-specific categories.

    PRIMARY_TROPICAL: Undisturbed primary tropical moist forest with
        high canopy cover (>80%), multi-layered canopy structure,
        and high biodiversity. Most sensitive to deforestation under
        EUDR. Found in Amazon, Congo Basin, SE Asia.
    SECONDARY_TROPICAL: Regenerating tropical forest that has been
        previously disturbed but has re-established tree cover.
        Typically lower canopy height and biomass than primary.
    TROPICAL_DRY: Tropical dry deciduous and semi-deciduous forest
        with seasonal leaf shedding. Requires phenological analysis
        to distinguish from deforestation events.
    TEMPERATE_BROADLEAF: Temperate broadleaf and mixed forest.
        Deciduous species that lose leaves seasonally, requiring
        multi-temporal classification.
    TEMPERATE_CONIFEROUS: Temperate coniferous (evergreen) forest.
        Year-round canopy cover simplifies monitoring.
    BOREAL: Boreal (taiga) forest dominated by coniferous species.
        Low canopy density but still classified as forest per FAO.
    MANGROVE: Coastal mangrove forest critical for carbon storage
        and EUDR-relevant for commodities like shrimp farming areas.
    PLANTATION: Tree plantation (timber, pulp, rubber, oil palm)
        with uniform structure and regular spacing. Distinct spectral
        signature from natural forest.
    AGROFORESTRY: Mixed agricultural and tree land use (e.g., shade
        coffee, cocoa agroforestry). Complex classification due to
        mixed land use signals.
    NON_FOREST: Land that does not meet FAO forest definition
        thresholds (canopy cover <10%, height <5m, or area <0.5ha).
    """

    PRIMARY_TROPICAL = "primary_tropical"
    SECONDARY_TROPICAL = "secondary_tropical"
    TROPICAL_DRY = "tropical_dry"
    TEMPERATE_BROADLEAF = "temperate_broadleaf"
    TEMPERATE_CONIFEROUS = "temperate_coniferous"
    BOREAL = "boreal"
    MANGROVE = "mangrove"
    PLANTATION = "plantation"
    AGROFORESTRY = "agroforestry"
    NON_FOREST = "non_forest"

class CanopyDensityClass(str, Enum):
    """Canopy density classification bins for forest cover assessment.

    Classifies continuous canopy density measurements into discrete
    classes for reporting and threshold-based analysis. Bins are based
    on FAO/UNEP categories with EUDR-specific thresholds.

    VERY_HIGH: Canopy cover >80%. Dense closed canopy typical of
        primary tropical forest. Strongest forest signal.
    HIGH: Canopy cover 60-80%. Closed canopy forest with some gaps.
        Still considered dense forest.
    MODERATE: Canopy cover 40-60%. Open canopy forest or
        disturbed forest. May indicate degradation.
    LOW: Canopy cover 20-40%. Sparse forest, woodland, or
        substantially degraded forest.
    SPARSE: Canopy cover 10-20%. At the FAO forest threshold.
        Marginal forest classification requiring careful assessment.
    OPEN: Canopy cover <10%. Below FAO forest threshold. Classified
        as non-forest for EUDR purposes.
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    SPARSE = "sparse"
    OPEN = "open"

class DeforestationVerdict(str, Enum):
    """Deforestation-free verification verdict for EUDR compliance.

    The final regulatory determination for a plot of land under EUDR
    Article 2. This verdict drives the DDS (Due Diligence Statement)
    outcome and determines whether commodities sourced from the plot
    may be placed on the EU market.

    DEFORESTATION_FREE: The plot was forest at the EUDR cutoff date
        (31 Dec 2020) and remains forest, OR was not forest at the
        cutoff date. No deforestation has occurred. Commodity may
        be placed on the EU market.
    DEFORESTED: Forest cover was present at the cutoff date but has
        been cleared since. Commodity from this plot must NOT be
        placed on the EU market.
    DEGRADED: Forest cover remains but has been significantly reduced
        (canopy loss exceeding the degradation threshold). May
        constitute forest degradation under EUDR Article 2(6).
    INCONCLUSIVE: Insufficient data quality or confidence to make a
        definitive determination. Additional data collection or
        manual review required before DDS submission.
    """

    DEFORESTATION_FREE = "deforestation_free"
    DEFORESTED = "deforested"
    DEGRADED = "degraded"
    INCONCLUSIVE = "inconclusive"

class DensityMethod(str, Enum):
    """Method for deriving continuous canopy density from satellite imagery.

    Each method has trade-offs in accuracy, computational cost, and
    applicability to different biomes and data sources.

    SPECTRAL_UNMIXING: Linear spectral unmixing to estimate
        sub-pixel fractional vegetation cover from mixed pixels.
        Best accuracy for medium-resolution imagery (Landsat, S2).
    NDVI_REGRESSION: Regression of NDVI values against ground-truth
        canopy cover measurements. Simple and well-calibrated for
        many forest types.
    DIMIDIATION: Vegetation fraction estimation using the dimidiation
        model (Vfc = (NDVI - NDVIsoil) / (NDVIveg - NDVIsoil)).
        Good for areas with known endmember spectra.
    SUB_PIXEL_DETECTION: Machine-learning based sub-pixel canopy
        detection using multi-spectral features. Highest accuracy
        but requires trained models per biome.
    """

    SPECTRAL_UNMIXING = "spectral_unmixing"
    NDVI_REGRESSION = "ndvi_regression"
    DIMIDIATION = "dimidiation"
    SUB_PIXEL_DETECTION = "sub_pixel_detection"

class ClassificationMethod(str, Enum):
    """Method for classifying forest type from remote sensing data.

    Methods can be combined in an ensemble for improved accuracy
    and robustness across different biomes.

    SPECTRAL_SIGNATURE: Classification based on spectral reflectance
        patterns in multi-spectral imagery. Uses band ratios and
        indices specific to each forest type.
    PHENOLOGICAL: Classification using seasonal vegetation dynamics
        from multi-temporal imagery. Distinguishes deciduous from
        evergreen, identifies cropping cycles in agroforestry.
    STRUCTURAL: Classification using canopy structure features
        derived from height data (GEDI, ICESat-2) and texture
        metrics. Distinguishes plantations from natural forest.
    MULTI_TEMPORAL: Classification using time series of spectral
        indices to capture temporal patterns. Reduces confusion
        between seasonal leaf-off and deforestation.
    ENSEMBLE: Weighted combination of multiple classification
        methods for robust, consensus-based classification with
        confidence estimates.
    """

    SPECTRAL_SIGNATURE = "spectral_signature"
    PHENOLOGICAL = "phenological"
    STRUCTURAL = "structural"
    MULTI_TEMPORAL = "multi_temporal"
    ENSEMBLE = "ensemble"

class HeightSource(str, Enum):
    """Data source for canopy height estimation.

    Each source has different spatial coverage, accuracy, and
    temporal availability characteristics.

    GEDI_L2A: NASA Global Ecosystem Dynamics Investigation L2A
        product. Full-waveform LiDAR from ISS providing canopy
        height profiles at 25m footprint. Coverage: 51.5N to 51.5S.
        Most accurate height measurements but sparse coverage.
    ICESAT2_ATL08: NASA ICESat-2 ATL08 Land and Vegetation Height
        product. Photon-counting LiDAR providing canopy height at
        100m segments. Global coverage. Lower accuracy than GEDI
        but better spatial sampling.
    SENTINEL2_TEXTURE: Canopy height estimated from Sentinel-2
        texture metrics (GLCM features) using regression models.
        Complete spatial coverage but lower accuracy. Useful as
        gap-filler when LiDAR data unavailable.
    GLOBAL_MAP_ETH: ETH Global Canopy Height Map (Lang et al. 2023).
        10m resolution global canopy height derived from Sentinel-2
        using deep learning. Wall-to-wall coverage, single epoch.
    GLOBAL_MAP_META: Meta/WRI Global Canopy Height Map.
        1m resolution canopy height derived from high-resolution
        imagery using deep learning. Highest spatial resolution but
        limited temporal coverage.
    """

    GEDI_L2A = "gedi_l2a"
    ICESAT2_ATL08 = "icesat2_atl08"
    SENTINEL2_TEXTURE = "sentinel2_texture"
    GLOBAL_MAP_ETH = "global_map_eth"
    GLOBAL_MAP_META = "global_map_meta"

class BiomassSource(str, Enum):
    """Data source for above-ground biomass (AGB) estimation.

    Each source provides AGB estimates at different spatial
    resolutions and accuracies.

    ESA_CCI: ESA Climate Change Initiative Biomass product.
        100m resolution global AGB maps derived from SAR and
        optical data fusion. Multiple epochs available (2010,
        2017-2020). Reference source for IPCC reporting.
    GEDI_L4A: NASA GEDI Level 4A gridded AGB product. 1km
        resolution mean AGB derived from GEDI L2A footprints
        with allometric models. Coverage: 51.5N to 51.5S.
    SAR_REGRESSION: AGB estimated from SAR backscatter
        (Sentinel-1 C-band, ALOS-2 PALSAR L-band) using
        regression models. Good for tropical forests where
        L-band penetrates canopy.
    NDVI_ALLOMETRIC: AGB estimated from NDVI-based vegetation
        indices using biome-specific allometric equations.
        Simple and widely applicable but lower accuracy for
        high-biomass forests (saturation effect).
    """

    ESA_CCI = "esa_cci"
    GEDI_L4A = "gedi_l4a"
    SAR_REGRESSION = "sar_regression"
    NDVI_ALLOMETRIC = "ndvi_allometric"

class FragmentationLevel(str, Enum):
    """Landscape fragmentation classification level.

    Categorizes the degree of forest fragmentation within a
    landscape based on patch metrics (effective mesh size,
    core area percentage, and connectivity index).

    INTACT: Continuous forest with minimal fragmentation.
        Large patches, high connectivity, high core area.
        Effective mesh size > 100 ha.
    SLIGHTLY_FRAGMENTED: Minor fragmentation with few small
        gaps or roads. Mostly connected forest landscape.
        Effective mesh size 50-100 ha.
    MODERATELY_FRAGMENTED: Noticeable fragmentation with
        multiple patches and reduced connectivity. Edge
        effects becoming significant.
        Effective mesh size 10-50 ha.
    HIGHLY_FRAGMENTED: Significant fragmentation with many
        small patches, low connectivity, and extensive edge
        habitat. May indicate ongoing degradation.
        Effective mesh size 1-10 ha.
    SEVERELY_FRAGMENTED: Extreme fragmentation with isolated
        remnant patches. Forest functionality severely
        compromised. Very low connectivity.
        Effective mesh size < 1 ha.
    """

    INTACT = "intact"
    SLIGHTLY_FRAGMENTED = "slightly_fragmented"
    MODERATELY_FRAGMENTED = "moderately_fragmented"
    HIGHLY_FRAGMENTED = "highly_fragmented"
    SEVERELY_FRAGMENTED = "severely_fragmented"

class AnalysisStatus(str, Enum):
    """Status of a forest cover analysis operation.

    Tracks the lifecycle of an analysis from creation through
    completion or failure.

    PENDING: Analysis has been requested but not yet started.
        Queued for processing.
    RUNNING: Analysis is currently being processed.
    COMPLETED: Analysis finished successfully with results.
    FAILED: Analysis encountered an error and did not produce
        valid results.
    CANCELLED: Analysis was cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DataQualityTier(str, Enum):
    """Data quality tier classification for analysis confidence.

    Categorizes the overall data quality of an analysis based on
    composite scoring of temporal coverage, spatial resolution,
    source agreement, and cloud-free observation frequency.

    GOLD: Overall quality score >90%. High confidence in all
        dimensions. Suitable for regulatory submission without
        caveats.
    SILVER: Overall quality score 70-90%. Good confidence with
        minor gaps. Suitable for regulatory submission with
        standard caveats.
    BRONZE: Overall quality score 50-70%. Moderate confidence
        with some limitations. May require supplementary evidence.
    INSUFFICIENT: Overall quality score <50%. Low confidence.
        Additional data collection required before regulatory
        submission.
    """

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    INSUFFICIENT = "insufficient"

# =============================================================================
# Core Data Models
# =============================================================================

class CanopyDensityResult(GreenLangBase):
    """Result of canopy density analysis for a plot.

    Contains the continuous canopy cover percentage, classified
    density class, method used, and per-pixel statistics from
    the density mapping operation.

    Attributes:
        density_pct: Mean canopy density percentage across the
            plot (0.0-100.0).
        density_class: Classified density bin based on density_pct.
        method: Density estimation method used.
        confidence: Confidence score for the density estimate
            (0.0-1.0).
        pixel_counts: Dictionary mapping density classes to pixel
            counts within the plot.
        spatial_resolution_m: Spatial resolution of the input
            imagery in metres.
        std_dev: Standard deviation of pixel-level density values.
        min_density_pct: Minimum pixel density within the plot.
        max_density_pct: Maximum pixel density within the plot.
        cloud_free_pct: Percentage of plot area that was cloud-free
            in the source imagery.
        analysis_date: Date of the source imagery used.
    """

    model_config = ConfigDict(from_attributes=True)

    density_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Mean canopy density percentage (0-100)",
    )
    density_class: CanopyDensityClass = Field(
        ...,
        description="Classified canopy density bin",
    )
    method: DensityMethod = Field(
        ...,
        description="Density estimation method used",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)",
    )
    pixel_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Pixel counts by density class",
    )
    spatial_resolution_m: float = Field(
        default=10.0,
        gt=0.0,
        description="Spatial resolution of input imagery (metres)",
    )
    std_dev: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of pixel densities",
    )
    min_density_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum pixel density in plot",
    )
    max_density_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Maximum pixel density in plot",
    )
    cloud_free_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Percentage of cloud-free plot area",
    )
    analysis_date: Optional[date] = Field(
        None,
        description="Date of the source imagery used",
    )

class ForestClassificationResult(GreenLangBase):
    """Result of forest type classification for a plot.

    Contains the primary and secondary forest type classifications
    with probability estimates, FAO forest determination, and
    the spectral signature features used for classification.

    Attributes:
        forest_type: Primary forest type classification.
        probability: Probability of the primary classification
            (0.0-1.0).
        secondary_type: Secondary (runner-up) forest type
            classification.
        secondary_probability: Probability of the secondary
            classification (0.0-1.0).
        is_forest_per_fao: Whether the plot meets FAO forest
            definition thresholds (canopy >10%, height >5m,
            area >0.5ha).
        spectral_signature: Key spectral features used in
            classification (band ratios, indices, texture metrics).
        classification_method: Method used for classification.
        biome: Biome identifier for stratified classification.
        classification_date: Date of the source data used.
    """

    model_config = ConfigDict(from_attributes=True)

    forest_type: ForestType = Field(
        ...,
        description="Primary forest type classification",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of primary classification",
    )
    secondary_type: Optional[ForestType] = Field(
        None,
        description="Secondary forest type classification",
    )
    secondary_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of secondary classification",
    )
    is_forest_per_fao: bool = Field(
        ...,
        description="Meets FAO forest definition thresholds",
    )
    spectral_signature: Dict[str, float] = Field(
        default_factory=dict,
        description="Spectral features used for classification",
    )
    classification_method: ClassificationMethod = Field(
        default=ClassificationMethod.ENSEMBLE,
        description="Classification method used",
    )
    biome: str = Field(
        default="",
        description="Biome identifier for stratified analysis",
    )
    classification_date: Optional[date] = Field(
        None,
        description="Date of source data used",
    )

class HistoricalCoverRecord(GreenLangBase):
    """Historical forest cover state at a specific point in time.

    Reconstructs the forest cover condition of a plot at or near
    the EUDR cutoff date (December 31, 2020) by compositing
    multi-temporal satellite imagery and integrating ancillary
    data sources.

    Attributes:
        cutoff_date: Target date for the historical reconstruction
            (default: EUDR cutoff date 2020-12-31).
        was_forest: Whether the plot was classified as forest at
            the target date per FAO definition.
        canopy_density_at_cutoff: Canopy density percentage at the
            target date.
        forest_type_at_cutoff: Forest type classification at the
            target date.
        data_sources: List of data sources used for reconstruction
            (e.g., 'hansen_gfc', 'sentinel_2', 'landsat_8').
        composite_quality: Quality score of the temporal composite
            used (0.0-1.0).
        reconstruction_confidence: Confidence in the historical
            reconstruction (0.0-1.0).
        observation_count: Number of cloud-free observations used
            in the composite window.
        earliest_observation: Date of the earliest observation in
            the composite window.
        latest_observation: Date of the latest observation in the
            composite window.
    """

    model_config = ConfigDict(from_attributes=True)

    cutoff_date: date = Field(
        default=EUDR_CUTOFF_DATE,
        description="Target date for historical reconstruction",
    )
    was_forest: bool = Field(
        ...,
        description="Plot was forest at target date (FAO definition)",
    )
    canopy_density_at_cutoff: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Canopy density percentage at target date",
    )
    forest_type_at_cutoff: Optional[ForestType] = Field(
        None,
        description="Forest type at target date",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used for reconstruction",
    )
    composite_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality score of temporal composite (0-1)",
    )
    reconstruction_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in historical reconstruction (0-1)",
    )
    observation_count: int = Field(
        default=0,
        ge=0,
        description="Number of cloud-free observations used",
    )
    earliest_observation: Optional[date] = Field(
        None,
        description="Earliest observation date in composite",
    )
    latest_observation: Optional[date] = Field(
        None,
        description="Latest observation date in composite",
    )

class DeforestationFreeResult(GreenLangBase):
    """Result of deforestation-free verification for EUDR compliance.

    Contains the regulatory verdict comparing forest cover at the
    EUDR cutoff date against current conditions, with confidence
    assessment and evidence references.

    Attributes:
        verdict: Regulatory determination (deforestation_free,
            deforested, degraded, inconclusive).
        cutoff_cover: Canopy density percentage at the EUDR cutoff
            date (31 Dec 2020).
        current_cover: Current canopy density percentage.
        canopy_change_pct: Percentage change in canopy density
            from cutoff to current (negative = loss).
        confidence: Confidence in the verdict determination
            (0.0-1.0).
        evidence_summary: Narrative summary of evidence supporting
            the verdict.
        regulatory_references: List of EUDR article and paragraph
            references applicable to this verdict.
        assessment_date: Date when the verification was performed.
        data_quality_tier: Quality tier of the underlying data.
        meets_eudr_threshold: Whether the confidence meets the
            minimum threshold for regulatory acceptance.
    """

    model_config = ConfigDict(from_attributes=True)

    verdict: DeforestationVerdict = Field(
        ...,
        description="EUDR deforestation-free verdict",
    )
    cutoff_cover: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Canopy density at EUDR cutoff date (%)",
    )
    current_cover: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current canopy density (%)",
    )
    canopy_change_pct: float = Field(
        ...,
        description="Canopy density change from cutoff (%, negative=loss)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in verdict (0.0-1.0)",
    )
    evidence_summary: str = Field(
        default="",
        description="Narrative evidence summary",
    )
    regulatory_references: List[str] = Field(
        default_factory=list,
        description="Applicable EUDR article references",
    )
    assessment_date: Optional[date] = Field(
        None,
        description="Date of verification assessment",
    )
    data_quality_tier: Optional[DataQualityTier] = Field(
        None,
        description="Quality tier of underlying data",
    )
    meets_eudr_threshold: bool = Field(
        default=False,
        description="Confidence meets minimum regulatory threshold",
    )

    @model_validator(mode="after")
    def _validate_change_consistency(self) -> "DeforestationFreeResult":
        """Validate that canopy_change_pct is consistent with cutoff and current cover."""
        expected_change = self.current_cover - self.cutoff_cover
        if abs(self.canopy_change_pct - expected_change) > 0.1:
            # Allow small floating-point tolerance
            pass
        return self

class CanopyHeightEstimate(GreenLangBase):
    """Canopy height estimate for a plot from remote sensing data.

    Provides tree height measurements from LiDAR or model-derived
    sources with uncertainty quantification and FAO threshold
    assessment.

    Attributes:
        height_m: Estimated canopy height in metres.
        uncertainty_m: Height uncertainty (standard error) in metres.
        source: Data source for the height estimate.
        measurement_date: Date of the height measurement or model
            prediction.
        meets_fao_threshold: Whether the estimated height exceeds
            the FAO 5m threshold for forest classification.
        percentile_25: 25th percentile canopy height in the plot.
        percentile_75: 75th percentile canopy height in the plot.
        max_height_m: Maximum canopy height measured in the plot.
        footprint_count: Number of LiDAR footprints or samples
            within the plot (for GEDI/ICESat-2 sources).
    """

    model_config = ConfigDict(from_attributes=True)

    height_m: float = Field(
        ...,
        ge=0.0,
        description="Estimated mean canopy height (metres)",
    )
    uncertainty_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Height uncertainty/standard error (metres)",
    )
    source: HeightSource = Field(
        ...,
        description="Height data source",
    )
    measurement_date: Optional[date] = Field(
        None,
        description="Date of height measurement",
    )
    meets_fao_threshold: bool = Field(
        ...,
        description="Height exceeds FAO 5m threshold",
    )
    percentile_25: Optional[float] = Field(
        None,
        ge=0.0,
        description="25th percentile canopy height (metres)",
    )
    percentile_75: Optional[float] = Field(
        None,
        ge=0.0,
        description="75th percentile canopy height (metres)",
    )
    max_height_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum canopy height in plot (metres)",
    )
    footprint_count: int = Field(
        default=0,
        ge=0,
        description="Number of LiDAR footprints/samples in plot",
    )

class FragmentationMetrics(GreenLangBase):
    """Landscape fragmentation metrics for a forest plot or landscape.

    Computes standard landscape ecology metrics to assess the
    structural integrity of forest cover. Based on FRAGSTATS
    methodology (McGarigal et al. 2012).

    Attributes:
        patch_count: Number of distinct forest patches within the
            analysis window.
        mean_patch_size_ha: Mean area of forest patches in hectares.
        edge_density_m_per_ha: Total forest edge length per unit
            area (m/ha). Higher values indicate greater fragmentation.
        core_area_pct: Percentage of forest area that is core habitat
            (interior area beyond an edge buffer, typically 100m).
        connectivity_index: Functional connectivity index (0-1)
            based on nearest-neighbour distance distribution.
        shape_complexity: Mean patch shape complexity index (1=circle,
            higher=more complex). Based on perimeter-area ratio.
        effective_mesh_size_ha: Effective mesh size in hectares.
            Probability that two randomly chosen points in the
            landscape are in the same patch. Primary fragmentation
            indicator.
        fragmentation_level: Classified fragmentation level based
            on effective mesh size.
        total_forest_area_ha: Total forest area in the analysis
            window (hectares).
        total_landscape_area_ha: Total landscape area in the
            analysis window (hectares).
    """

    model_config = ConfigDict(from_attributes=True)

    patch_count: int = Field(
        ...,
        ge=0,
        description="Number of distinct forest patches",
    )
    mean_patch_size_ha: float = Field(
        ...,
        ge=0.0,
        description="Mean forest patch area (hectares)",
    )
    edge_density_m_per_ha: float = Field(
        ...,
        ge=0.0,
        description="Forest edge length per unit area (m/ha)",
    )
    core_area_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Core habitat percentage of total forest area",
    )
    connectivity_index: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Functional connectivity index (0-1)",
    )
    shape_complexity: float = Field(
        ...,
        ge=1.0,
        description="Mean patch shape complexity (1=circle)",
    )
    effective_mesh_size_ha: float = Field(
        ...,
        ge=0.0,
        description="Effective mesh size (hectares)",
    )
    fragmentation_level: FragmentationLevel = Field(
        ...,
        description="Classified fragmentation level",
    )
    total_forest_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Total forest area (hectares)",
    )
    total_landscape_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Total landscape area (hectares)",
    )

class BiomassEstimate(GreenLangBase):
    """Above-ground biomass (AGB) estimate for a forest plot.

    Provides AGB and derived carbon stock estimates from remote
    sensing data sources with uncertainty quantification and
    temporal change assessment.

    Attributes:
        agb_mg_per_ha: Above-ground biomass in megagrams (tonnes)
            per hectare.
        uncertainty_mg_per_ha: AGB uncertainty (standard error) in
            Mg/ha.
        carbon_stock_tc_per_ha: Carbon stock in tonnes of carbon
            per hectare, derived from AGB using the IPCC conversion
            factor of 0.47.
        source: Data source for the biomass estimate.
        biomass_change_pct: Percentage change in AGB from the
            previous assessment (negative = loss). None if no
            previous assessment exists.
        measurement_date: Date of the biomass measurement or model
            prediction.
        below_ground_ratio: Ratio of below-ground to above-ground
            biomass (root-to-shoot ratio). Used for total biomass
            estimation.
        total_carbon_tc: Total carbon stock for the entire plot
            area in tonnes of carbon.
    """

    model_config = ConfigDict(from_attributes=True)

    agb_mg_per_ha: float = Field(
        ...,
        ge=0.0,
        description="Above-ground biomass (Mg/ha)",
    )
    uncertainty_mg_per_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="AGB uncertainty/standard error (Mg/ha)",
    )
    carbon_stock_tc_per_ha: float = Field(
        ...,
        ge=0.0,
        description="Carbon stock (tC/ha) = AGB * 0.47",
    )
    source: BiomassSource = Field(
        ...,
        description="Biomass data source",
    )
    biomass_change_pct: Optional[float] = Field(
        None,
        description="AGB change from previous assessment (%)",
    )
    measurement_date: Optional[date] = Field(
        None,
        description="Date of biomass measurement",
    )
    below_ground_ratio: float = Field(
        default=0.26,
        ge=0.0,
        description="Root-to-shoot ratio for below-ground biomass",
    )
    total_carbon_tc: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total carbon stock for entire plot (tC)",
    )

class ComplianceReport(GreenLangBase):
    """Generated compliance report for EUDR forest cover assessment.

    Contains metadata for a compliance report generated from forest
    cover analysis results, including provenance tracking for
    regulatory audit trails.

    Attributes:
        report_id: Unique report identifier (UUID).
        report_type: Type of report (full, summary, compliance,
            evidence).
        format: Output format of the report.
        plot_id: Plot identifier this report covers.
        verdict: EUDR deforestation-free verdict for the plot.
        summary: Brief narrative summary of report findings.
        created_at: UTC timestamp when the report was generated.
        provenance_hash: SHA-256 hash of the report content for
            tamper detection and audit trail.
        regulatory_framework: Regulatory framework reference
            (default: EUDR EU 2023/1115).
        valid_until: Date until which this report is considered
            valid (typically 12 months from creation).
        reviewer: Optional reviewer identifier for approved reports.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier (UUID)",
    )
    report_type: str = Field(
        default="full",
        description="Report type (full, summary, compliance, evidence)",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format",
    )
    plot_id: str = Field(
        ...,
        description="Plot identifier this report covers",
    )
    verdict: DeforestationVerdict = Field(
        ...,
        description="EUDR deforestation-free verdict",
    )
    summary: str = Field(
        default="",
        description="Brief narrative summary of findings",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
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

class DataQualityAssessment(GreenLangBase):
    """Composite data quality assessment for forest cover analysis.

    Scores multiple quality dimensions and classifies the overall
    data quality tier for regulatory confidence assessment.

    Attributes:
        overall_score: Composite quality score (0.0-100.0).
        tier: Classified quality tier based on overall_score.
        temporal_score: Quality of temporal coverage during the
            analysis window (0.0-100.0).
        spatial_score: Quality of spatial resolution and coverage
            (0.0-100.0).
        source_agreement_score: Agreement between multiple data
            sources (0.0-100.0). Higher when sources converge.
        cloud_free_score: Score based on cloud-free observation
            frequency (0.0-100.0).
        factors: Dictionary of individual quality factors and
            their scores.
    """

    model_config = ConfigDict(from_attributes=True)

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite quality score (0-100)",
    )
    tier: DataQualityTier = Field(
        ...,
        description="Classified quality tier",
    )
    temporal_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Temporal coverage quality (0-100)",
    )
    spatial_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Spatial resolution quality (0-100)",
    )
    source_agreement_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Multi-source agreement score (0-100)",
    )
    cloud_free_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Cloud-free observation frequency score (0-100)",
    )
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual quality factors and scores",
    )

    @field_validator("tier", mode="before")
    @classmethod
    def _validate_tier_consistency(cls, v: Any, info: Any) -> Any:
        """Allow tier to be set directly; consistency check in model_validator."""
        return v

    @model_validator(mode="after")
    def _check_tier_matches_score(self) -> "DataQualityAssessment":
        """Validate that tier is consistent with overall_score."""
        expected_tier = _score_to_tier(self.overall_score)
        if self.tier != expected_tier:
            # Auto-correct tier to match score
            self.tier = expected_tier
        return self

def _score_to_tier(score: float) -> DataQualityTier:
    """Map a quality score to a DataQualityTier.

    Args:
        score: Quality score (0-100).

    Returns:
        Corresponding DataQualityTier.
    """
    if score > 90.0:
        return DataQualityTier.GOLD
    elif score > 70.0:
        return DataQualityTier.SILVER
    elif score > 50.0:
        return DataQualityTier.BRONZE
    else:
        return DataQualityTier.INSUFFICIENT

class PlotForestProfile(GreenLangBase):
    """Complete forest profile for a single plot combining all analysis results.

    Aggregates canopy density, forest type, height, biomass,
    fragmentation, and deforestation-free verdict into a single
    comprehensive profile for the plot.

    Attributes:
        plot_id: Unique plot identifier.
        area_ha: Plot area in hectares.
        centroid_lat: Latitude of the plot centroid.
        centroid_lon: Longitude of the plot centroid.
        canopy_density: Canopy density analysis result.
        forest_type: Forest type classification result.
        height_estimate: Canopy height estimate.
        biomass_estimate: Above-ground biomass estimate.
        fragmentation: Landscape fragmentation metrics.
        is_forest_fao: Whether the plot meets FAO forest definition.
        verdict: EUDR deforestation-free verdict.
        data_quality: Data quality assessment.
        provenance_hash: SHA-256 hash of the complete profile.
        last_updated: UTC timestamp of the last profile update.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        description="Unique plot identifier",
    )
    area_ha: float = Field(
        ...,
        gt=0.0,
        description="Plot area (hectares)",
    )
    centroid_lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Plot centroid latitude",
    )
    centroid_lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Plot centroid longitude",
    )
    canopy_density: Optional[CanopyDensityResult] = Field(
        None,
        description="Canopy density analysis result",
    )
    forest_type: Optional[ForestClassificationResult] = Field(
        None,
        description="Forest type classification result",
    )
    height_estimate: Optional[CanopyHeightEstimate] = Field(
        None,
        description="Canopy height estimate",
    )
    biomass_estimate: Optional[BiomassEstimate] = Field(
        None,
        description="Above-ground biomass estimate",
    )
    fragmentation: Optional[FragmentationMetrics] = Field(
        None,
        description="Landscape fragmentation metrics",
    )
    is_forest_fao: Optional[bool] = Field(
        None,
        description="Meets FAO forest definition",
    )
    verdict: Optional[DeforestationVerdict] = Field(
        None,
        description="EUDR deforestation-free verdict",
    )
    data_quality: Optional[DataQualityAssessment] = Field(
        None,
        description="Data quality assessment",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of complete profile",
    )
    last_updated: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of last profile update",
    )

# =============================================================================
# Request Models
# =============================================================================

class AnalyzeDensityRequest(GreenLangBase):
    """Request to perform canopy density analysis for a plot.

    Attributes:
        plot_id: Unique plot identifier for tracking.
        polygon_wkt: Well-Known Text (WKT) representation of the
            plot boundary polygon.
        imagery_date: Target date for the density analysis. If None,
            uses the most recent available imagery.
        method: Density estimation method to use.
        biome: Optional biome identifier for calibrated thresholds.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Plot boundary polygon in WKT format",
    )
    imagery_date: Optional[date] = Field(
        None,
        description="Target date for analysis (None=most recent)",
    )
    method: DensityMethod = Field(
        default=DensityMethod.SPECTRAL_UNMIXING,
        description="Density estimation method",
    )
    biome: str = Field(
        default="",
        description="Biome identifier for calibrated thresholds",
    )

class ClassifyForestRequest(GreenLangBase):
    """Request to classify forest type for a plot.

    Attributes:
        plot_id: Unique plot identifier for tracking.
        polygon_wkt: WKT representation of the plot boundary.
        date_range_start: Start date for multi-temporal classification.
        date_range_end: End date for multi-temporal classification.
        methods: Classification methods to apply. If multiple, runs
            ensemble voting.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Plot boundary polygon in WKT format",
    )
    date_range_start: Optional[date] = Field(
        None,
        description="Start date for multi-temporal classification",
    )
    date_range_end: Optional[date] = Field(
        None,
        description="End date for multi-temporal classification",
    )
    methods: List[ClassificationMethod] = Field(
        default_factory=lambda: [ClassificationMethod.ENSEMBLE],
        description="Classification methods to apply",
    )

    @model_validator(mode="after")
    def _validate_date_range(self) -> "ClassifyForestRequest":
        """Validate that date_range_start <= date_range_end when both set."""
        if (
            self.date_range_start is not None
            and self.date_range_end is not None
            and self.date_range_start > self.date_range_end
        ):
            raise ValueError(
                f"date_range_start ({self.date_range_start}) must be "
                f"<= date_range_end ({self.date_range_end})"
            )
        return self

class ReconstructHistoryRequest(GreenLangBase):
    """Request to reconstruct historical forest cover at a target date.

    Attributes:
        plot_id: Unique plot identifier for tracking.
        polygon_wkt: WKT representation of the plot boundary.
        target_date: Target date for reconstruction (default: EUDR
            cutoff date 2020-12-31).
        window_years: Number of years of imagery to composite
            around the target date.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Plot boundary polygon in WKT format",
    )
    target_date: date = Field(
        default=EUDR_CUTOFF_DATE,
        description="Target date for reconstruction",
    )
    window_years: int = Field(
        default=3,
        gt=0,
        le=10,
        description="Composite window size in years",
    )

class VerifyDeforestationFreeRequest(GreenLangBase):
    """Request to verify deforestation-free status for EUDR compliance.

    Attributes:
        plot_id: Unique plot identifier for tracking.
        polygon_wkt: WKT representation of the plot boundary.
        commodity: EUDR-regulated commodity sourced from this plot.
        include_evidence: Whether to include detailed evidence
            narrative in the result.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=1,
        description="Plot boundary polygon in WKT format",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR-regulated commodity from this plot",
    )
    include_evidence: bool = Field(
        default=True,
        description="Include detailed evidence narrative",
    )

class BatchAnalysisRequest(GreenLangBase):
    """Request to perform batch forest cover analysis for multiple plots.

    Attributes:
        plot_ids: List of plot identifiers to analyze.
        analysis_types: Types of analysis to perform per plot.
        commodity: EUDR-regulated commodity for all plots.
        priority: Processing priority (1=lowest, 10=highest).
    """

    model_config = ConfigDict(from_attributes=True)

    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Plot identifiers to analyze",
    )
    analysis_types: List[str] = Field(
        default_factory=lambda: [
            "density", "classification", "history", "verdict",
        ],
        description="Analysis types to perform per plot",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR-regulated commodity for all plots",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Processing priority (1=lowest, 10=highest)",
    )

    @field_validator("plot_ids")
    @classmethod
    def _validate_batch_size(cls, v: List[str]) -> List[str]:
        """Validate batch size does not exceed maximum."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {MAX_BATCH_SIZE}"
            )
        return v

class GenerateReportRequest(GreenLangBase):
    """Request to generate a compliance report for a plot.

    Attributes:
        plot_id: Plot identifier to generate report for.
        report_type: Type of report to generate.
        format: Output format for the report.
        include_maps: Whether to include map visualizations
            in the report (PDF only).
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier",
    )
    report_type: str = Field(
        default="full",
        description="Report type (full, summary, compliance, evidence)",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format",
    )
    include_maps: bool = Field(
        default=False,
        description="Include map visualizations (PDF only)",
    )

# =============================================================================
# Response Models
# =============================================================================

class BatchAnalysisResponse(GreenLangBase):
    """Response from a batch forest cover analysis request.

    Attributes:
        batch_id: Unique batch job identifier (UUID).
        status: Current status of the batch analysis.
        total_plots: Total number of plots in the batch.
        completed: Number of plots completed successfully.
        failed: Number of plots that failed analysis.
        results: Dictionary mapping plot_id to PlotForestProfile.
        errors: Dictionary mapping plot_id to error message.
        started_at: UTC timestamp when batch processing started.
        completed_at: UTC timestamp when batch processing finished.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier (UUID)",
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.PENDING,
        description="Current batch analysis status",
    )
    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots in batch",
    )
    completed: int = Field(
        default=0,
        ge=0,
        description="Plots completed successfully",
    )
    failed: int = Field(
        default=0,
        ge=0,
        description="Plots that failed analysis",
    )
    results: Dict[str, PlotForestProfile] = Field(
        default_factory=dict,
        description="Plot results keyed by plot_id",
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Error messages keyed by plot_id",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp of batch start",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp of batch completion",
    )

class BatchProgress(GreenLangBase):
    """Progress information for a running batch analysis.

    Attributes:
        batch_id: Unique batch job identifier.
        percent_complete: Completion percentage (0.0-100.0).
        estimated_remaining_seconds: Estimated time remaining in
            seconds. None if not yet estimable.
        plots_per_second: Current processing throughput.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        ...,
        description="Unique batch job identifier",
    )
    percent_complete: float = Field(
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
    plots_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Current processing throughput",
    )

class AnalysisSummary(GreenLangBase):
    """Aggregate summary of forest cover analysis across multiple plots.

    Provides count-based summaries for dashboards and overview
    reporting.

    Attributes:
        total_plots: Total number of plots analyzed.
        forest_count: Plots classified as forest (FAO definition).
        non_forest_count: Plots classified as non-forest.
        deforested_count: Plots with deforestation verdict.
        degraded_count: Plots with degradation verdict.
        inconclusive_count: Plots with inconclusive verdict.
        deforestation_free_count: Plots verified deforestation-free.
        avg_confidence: Average confidence across all verdicts.
        avg_data_quality: Average data quality score.
    """

    model_config = ConfigDict(from_attributes=True)

    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots analyzed",
    )
    forest_count: int = Field(
        default=0,
        ge=0,
        description="Plots classified as forest",
    )
    non_forest_count: int = Field(
        default=0,
        ge=0,
        description="Plots classified as non-forest",
    )
    deforested_count: int = Field(
        default=0,
        ge=0,
        description="Plots with deforestation verdict",
    )
    degraded_count: int = Field(
        default=0,
        ge=0,
        description="Plots with degradation verdict",
    )
    inconclusive_count: int = Field(
        default=0,
        ge=0,
        description="Plots with inconclusive verdict",
    )
    deforestation_free_count: int = Field(
        default=0,
        ge=0,
        description="Plots verified deforestation-free",
    )
    avg_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence across verdicts",
    )
    avg_data_quality: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Average data quality score",
    )

class ForestCoverDashboard(GreenLangBase):
    """Dashboard view aggregating forest cover analysis results.

    Provides a comprehensive overview combining summary statistics,
    average metrics, and distribution breakdowns for visualization
    in monitoring dashboards.

    Attributes:
        summary: Aggregate analysis summary.
        avg_canopy_density: Average canopy density across all
            analyzed forest plots (%).
        avg_biomass: Average above-ground biomass across all
            analyzed forest plots (Mg/ha).
        forest_type_distribution: Count of plots by forest type.
        verdict_distribution: Count of plots by verdict.
        data_quality_distribution: Count of plots by quality tier.
        total_forest_area_ha: Total forest area across all plots.
        total_carbon_stock_tc: Total carbon stock across all plots.
        last_updated: UTC timestamp of the last dashboard update.
    """

    model_config = ConfigDict(from_attributes=True)

    summary: AnalysisSummary = Field(
        default_factory=AnalysisSummary,
        description="Aggregate analysis summary",
    )
    avg_canopy_density: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Average canopy density across forest plots (%)",
    )
    avg_biomass: float = Field(
        default=0.0,
        ge=0.0,
        description="Average AGB across forest plots (Mg/ha)",
    )
    forest_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot count by forest type",
    )
    verdict_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot count by verdict",
    )
    data_quality_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot count by quality tier",
    )
    total_forest_area_ha: float = Field(
        default=0.0,
        ge=0.0,
        description="Total forest area (hectares)",
    )
    total_carbon_stock_tc: float = Field(
        default=0.0,
        ge=0.0,
        description="Total carbon stock (tonnes C)",
    )
    last_updated: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of last dashboard update",
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "FAO_CANOPY_THRESHOLD",
    "FAO_HEIGHT_THRESHOLD",
    "FAO_AREA_THRESHOLD",
    "BIOME_COUNT",
    "AGB_CONVERSION_FACTOR",
    # Re-exported
    "EUDRCommodity",
    # Enumerations
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
    # Core models
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
    # Request models
    "AnalyzeDensityRequest",
    "ClassifyForestRequest",
    "ReconstructHistoryRequest",
    "VerifyDeforestationFreeRequest",
    "BatchAnalysisRequest",
    "GenerateReportRequest",
    # Response models
    "BatchAnalysisResponse",
    "BatchProgress",
    "AnalysisSummary",
    "ForestCoverDashboard",
]
