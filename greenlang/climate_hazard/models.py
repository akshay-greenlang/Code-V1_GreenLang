# -*- coding: utf-8 -*-
"""
Climate Hazard Connector Service Data Models - AGENT-DATA-020

Pydantic v2 data models for the Climate Hazard Connector SDK. Attempts to
re-export Layer 1 engines and classes from the GIS/Mapping Connector
(SpatialAnalyzerEngine, BoundaryResolverEngine, CRSTransformerEngine),
and defines all SDK models for climate hazard data ingestion, risk indexing,
scenario projections, asset exposure, vulnerability scoring, compound
hazards, compliance reporting, and pipeline orchestration.

Re-exported Layer 1 sources (best-effort, with fallback stubs):
    - greenlang.gis_connector.spatial_analyzer: SpatialAnalyzerEngine
    - greenlang.gis_connector.boundary_resolver: BoundaryResolverEngine
    - greenlang.gis_connector.crs_transformer: CRSTransformerEngine

New enumerations (12):
    - HazardType, RiskLevel, Scenario, TimeHorizon, AssetType,
      ReportType, ReportFormat, DataSourceType, ExposureLevel,
      SensitivityLevel, AdaptiveCapacity, VulnerabilityLevel

New SDK models (14):
    - Location, HazardSource, HazardDataRecord, HazardEvent, RiskIndex,
      ScenarioProjection, Asset, ExposureResult, SensitivityProfile,
      AdaptiveCapacityProfile, VulnerabilityScore, CompoundHazard,
      ComplianceReport, PipelineRun

Request models (8):
    - RegisterSourceRequest, IngestDataRequest, CalculateRiskRequest,
      ProjectScenarioRequest, RegisterAssetRequest, AssessExposureRequest,
      ScoreVulnerabilityRequest, GenerateReportRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Layer 1 Re-exports (best-effort with stubs on ImportError)
# ---------------------------------------------------------------------------

try:
    from greenlang.gis_connector.spatial_analyzer import (  # type: ignore[import]
        SpatialAnalyzerEngine as L1SpatialAnalyzerEngine,
    )

    SpatialAnalyzerEngine = L1SpatialAnalyzerEngine
    _SAE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SAE_AVAILABLE = False

    class SpatialAnalyzerEngine:  # type: ignore[no-redef]
        """Stub re-export when gis_connector.spatial_analyzer is unavailable."""

        pass


try:
    from greenlang.gis_connector.boundary_resolver import (  # type: ignore[import]
        BoundaryResolverEngine as L1BoundaryResolverEngine,
    )

    BoundaryResolverEngine = L1BoundaryResolverEngine
    _BRE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BRE_AVAILABLE = False

    class BoundaryResolverEngine:  # type: ignore[no-redef]
        """Stub re-export when gis_connector.boundary_resolver is unavailable."""

        pass


try:
    from greenlang.gis_connector.crs_transformer import (  # type: ignore[import]
        CRSTransformerEngine as L1CRSTransformerEngine,
    )

    CRSTransformerEngine = L1CRSTransformerEngine
    _CTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CTE_AVAILABLE = False

    class CRSTransformerEngine:  # type: ignore[no-redef]
        """Stub re-export when gis_connector.crs_transformer is unavailable."""

        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of hazard data sources per tenant / namespace.
MAX_SOURCES_PER_NAMESPACE: int = 1_000

#: Maximum number of hazard data records in a single ingestion batch.
MAX_RECORDS_PER_BATCH: int = 100_000

#: Maximum number of assets per tenant / namespace.
MAX_ASSETS_PER_NAMESPACE: int = 50_000

#: Maximum number of hazard types in a compound hazard definition.
MAX_COMPOUND_HAZARDS: int = 12

#: Maximum number of sensitivity or adaptive-capacity factors per profile.
MAX_FACTORS_PER_PROFILE: int = 100

#: Default batch size for pipeline processing operations.
DEFAULT_PIPELINE_BATCH_SIZE: int = 1_000

#: Default confidence threshold for risk score classification.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.8

#: Maximum number of scenario projections per hazard-location pair.
MAX_PROJECTIONS_PER_PAIR: int = 50

#: Maximum search radius in kilometres for proximity-based exposure analysis.
MAX_SEARCH_RADIUS_KM: float = 500.0

#: Risk level boundaries (upper bounds, exclusive) for score classification.
RISK_LEVEL_BOUNDARIES: Dict[str, float] = {
    "negligible": 20.0,
    "low": 40.0,
    "medium": 60.0,
    "high": 80.0,
    "extreme": 100.0,
}

#: Supported IPCC climate scenario identifiers for CMIP6 and CMIP5.
SUPPORTED_SCENARIOS: tuple = (
    "ssp1_1_9",
    "ssp1_2_6",
    "ssp2_4_5",
    "ssp3_7_0",
    "ssp5_8_5",
    "rcp2_6",
    "rcp4_5",
    "rcp8_5",
)

#: Supported report output formats.
SUPPORTED_REPORT_FORMATS: tuple = ("json", "html", "markdown", "text", "csv")

#: Compliance frameworks supported by the reporting engine.
SUPPORTED_FRAMEWORKS: tuple = ("tcfd", "csrd_esrs", "sec_climate", "ifrs_s2", "ngfs")

#: Time horizon year ranges for climate projections.
TIME_HORIZON_RANGES: Dict[str, tuple] = {
    "baseline": (1995, 2014),
    "near_term": (2021, 2040),
    "mid_term": (2041, 2060),
    "long_term": (2061, 2080),
    "end_century": (2081, 2100),
}


# =============================================================================
# Enumerations (12)
# =============================================================================


class HazardType(str, Enum):
    """Classification of physical climate hazards.

    Enumerates the acute and chronic climate-related hazards supported
    by the Climate Hazard Connector for physical risk assessment.
    Hazard types align with TCFD physical risk taxonomy and IPCC AR6
    classifications.

    RIVERINE_FLOOD: Flooding caused by rivers overflowing their banks.
    COASTAL_FLOOD: Flooding caused by storm surge, tides, or wave action.
    DROUGHT: Extended period of abnormally low precipitation.
    EXTREME_HEAT: Temperature exceeding historical norms for the region.
    EXTREME_COLD: Temperature dropping below historical norms for the region.
    WILDFIRE: Uncontrolled fire in vegetation, forests, or grasslands.
    TROPICAL_CYCLONE: Rotating low-pressure weather system (hurricane/typhoon).
    EXTREME_PRECIPITATION: Rainfall or snowfall exceeding historical norms.
    WATER_STRESS: Chronic imbalance between water demand and supply.
    SEA_LEVEL_RISE: Long-term increase in mean sea level from thermal expansion.
    LANDSLIDE: Downslope movement of rock, earth, or debris from instability.
    COASTAL_EROSION: Loss of coastal land due to wave action and sea level rise.
    """

    RIVERINE_FLOOD = "riverine_flood"
    COASTAL_FLOOD = "coastal_flood"
    DROUGHT = "drought"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    WILDFIRE = "wildfire"
    TROPICAL_CYCLONE = "tropical_cyclone"
    EXTREME_PRECIPITATION = "extreme_precipitation"
    WATER_STRESS = "water_stress"
    SEA_LEVEL_RISE = "sea_level_rise"
    LANDSLIDE = "landslide"
    COASTAL_EROSION = "coastal_erosion"


class RiskLevel(str, Enum):
    """Qualitative risk classification derived from a numeric risk score.

    Maps continuous risk scores (0-100) to discrete levels for
    communication and decision-making. Thresholds follow industry
    conventions from TCFD and NGFS guidance.

    NEGLIGIBLE: Risk score 0-20; minimal expected impact.
    LOW: Risk score 20-40; limited impact unlikely to require action.
    MEDIUM: Risk score 40-60; moderate impact requiring monitoring.
    HIGH: Risk score 60-80; significant impact requiring adaptation.
    EXTREME: Risk score 80-100; critical impact requiring immediate action.
    """

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class Scenario(str, Enum):
    """IPCC climate scenario pathways for forward-looking projections.

    Covers both CMIP6 Shared Socioeconomic Pathways (SSP) and legacy
    CMIP5 Representative Concentration Pathways (RCP) to support
    transition from older to newer scenario frameworks.

    SSP1_1_9: Sustainability; very low emissions; 1.5C target.
    SSP1_2_6: Sustainability; low emissions; well-below 2C.
    SSP2_4_5: Middle of the road; intermediate emissions.
    SSP3_7_0: Regional rivalry; high emissions.
    SSP5_8_5: Fossil-fuelled development; very high emissions.
    RCP2_6: Legacy CMIP5 low emissions pathway (approx. 2C).
    RCP4_5: Legacy CMIP5 intermediate emissions pathway.
    RCP8_5: Legacy CMIP5 high emissions pathway (business-as-usual).
    """

    SSP1_1_9 = "ssp1_1_9"
    SSP1_2_6 = "ssp1_2_6"
    SSP2_4_5 = "ssp2_4_5"
    SSP3_7_0 = "ssp3_7_0"
    SSP5_8_5 = "ssp5_8_5"
    RCP2_6 = "rcp2_6"
    RCP4_5 = "rcp4_5"
    RCP8_5 = "rcp8_5"


class TimeHorizon(str, Enum):
    """Climate projection time horizons aligned with IPCC AR6 conventions.

    Defines the temporal windows used for climate hazard projections
    and scenario analysis. Each horizon maps to a specific year range
    used for data aggregation and risk scoring.

    BASELINE: 1995-2014; IPCC AR6 reference period for historical data.
    NEAR_TERM: 2021-2040; short-range projection window.
    MID_TERM: 2041-2060; medium-range projection window.
    LONG_TERM: 2061-2080; long-range projection window.
    END_CENTURY: 2081-2100; end-of-century projection window.
    """

    BASELINE = "baseline"
    NEAR_TERM = "near_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    END_CENTURY = "end_century"


class AssetType(str, Enum):
    """Classification of physical and operational assets subject to risk.

    Categorises assets registered in the Climate Hazard Connector
    for exposure and vulnerability analysis. Asset types determine
    which hazard-specific sensitivity factors and adaptive capacity
    indicators apply.

    FACILITY: Manufacturing plant, warehouse, office, or data centre.
    SUPPLY_CHAIN_NODE: Supplier site, logistics hub, or distribution centre.
    AGRICULTURAL_PLOT: Farmland, plantation, or forestry concession.
    INFRASTRUCTURE: Road, bridge, port, rail, or utility network segment.
    REAL_ESTATE: Commercial or residential property holding.
    NATURAL_ASSET: Protected area, wetland, or ecosystem service zone.
    WATER_SOURCE: Reservoir, aquifer, river intake, or desalination plant.
    COASTAL_ASSET: Harbour, seawall, coastal facility, or offshore platform.
    """

    FACILITY = "facility"
    SUPPLY_CHAIN_NODE = "supply_chain_node"
    AGRICULTURAL_PLOT = "agricultural_plot"
    INFRASTRUCTURE = "infrastructure"
    REAL_ESTATE = "real_estate"
    NATURAL_ASSET = "natural_asset"
    WATER_SOURCE = "water_source"
    COASTAL_ASSET = "coastal_asset"


class ReportType(str, Enum):
    """Type of climate risk report to generate.

    Determines the structure, content granularity, and audience
    for the generated climate risk report. Each type maps to a
    specific reporting template and compliance framework alignment.

    PHYSICAL_RISK_ASSESSMENT: Comprehensive hazard-by-hazard risk analysis.
    SCENARIO_ANALYSIS: Forward-looking scenario comparison (TCFD/NGFS).
    ADAPTATION_SCREENING: Adaptation measures screening and prioritisation.
    EXPOSURE_SUMMARY: Portfolio-level exposure and concentration summary.
    EXECUTIVE_DASHBOARD: High-level KPI dashboard for board reporting.
    """

    PHYSICAL_RISK_ASSESSMENT = "physical_risk_assessment"
    SCENARIO_ANALYSIS = "scenario_analysis"
    ADAPTATION_SCREENING = "adaptation_screening"
    EXPOSURE_SUMMARY = "exposure_summary"
    EXECUTIVE_DASHBOARD = "executive_dashboard"


class ReportFormat(str, Enum):
    """Output format for a climate risk report.

    JSON: Structured JSON for programmatic consumption and integration.
    HTML: Self-contained HTML page with formatting, maps, and charts.
    MARKDOWN: Markdown-formatted report for documentation systems.
    TEXT: Plain-text summary for terminal or log output.
    CSV: Comma-separated values for spreadsheet import and analysis.
    """

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    CSV = "csv"


class DataSourceType(str, Enum):
    """Classification of external data sources for hazard information.

    Categorises the provenance and methodology of hazard data
    ingested into the Climate Hazard Connector. Source type drives
    data quality expectations, update frequency, and spatial resolution.

    GLOBAL_DATABASE: Multi-hazard global dataset (e.g., ThinkHazard, Aqueduct).
    REGIONAL_INDEX: Regional or national hazard index (e.g., INFORM, EM-DAT).
    EVENT_CATALOG: Historical event catalogue (e.g., EM-DAT, DesInventar).
    SCENARIO_MODEL: Climate model output (e.g., CMIP6, CORDEX downscaled).
    SATELLITE: Remote sensing derived hazard data (e.g., MODIS fire, Sentinel).
    REANALYSIS: Reanalysis dataset (e.g., ERA5, MERRA-2, JRA-55).
    """

    GLOBAL_DATABASE = "global_database"
    REGIONAL_INDEX = "regional_index"
    EVENT_CATALOG = "event_catalog"
    SCENARIO_MODEL = "scenario_model"
    SATELLITE = "satellite"
    REANALYSIS = "reanalysis"


class ExposureLevel(str, Enum):
    """Qualitative exposure classification for an asset to a hazard.

    Describes the degree to which an asset is physically exposed
    to a specific climate hazard based on proximity, intensity at
    location, and frequency of occurrence.

    NONE: No detectable exposure within the analysis radius.
    LOW: Marginal exposure; hazard rarely reaches the asset location.
    MODERATE: Noticeable exposure; hazard periodically affects the area.
    HIGH: Significant exposure; hazard frequently impacts the location.
    CRITICAL: Extreme exposure; asset is in a primary impact zone.
    """

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SensitivityLevel(str, Enum):
    """Qualitative sensitivity classification for an entity to a hazard.

    Describes how strongly an asset or system is affected when
    exposed to a climate hazard. Sensitivity depends on physical
    characteristics, operational dependencies, and sectoral factors.

    VERY_LOW: Minimal sensitivity; operations largely unaffected.
    LOW: Limited sensitivity; minor operational disruption expected.
    MODERATE: Moderate sensitivity; notable disruption likely.
    HIGH: High sensitivity; significant disruption or damage expected.
    VERY_HIGH: Maximum sensitivity; severe damage or total loss likely.
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AdaptiveCapacity(str, Enum):
    """Qualitative adaptive capacity classification for an entity.

    Describes the ability of an asset, organisation, or system to
    adjust to climate hazards, moderate potential damages, take
    advantage of opportunities, or cope with consequences.

    VERY_LOW: Minimal adaptive capacity; no contingency measures.
    LOW: Limited adaptive capacity; basic contingency only.
    MODERATE: Moderate adaptive capacity; some adaptation in place.
    HIGH: Strong adaptive capacity; robust adaptation measures deployed.
    VERY_HIGH: Maximum adaptive capacity; comprehensive resilience programme.
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class VulnerabilityLevel(str, Enum):
    """Composite vulnerability classification combining exposure, sensitivity,
    and adaptive capacity.

    Follows the IPCC AR5/AR6 vulnerability framework where
    vulnerability = f(exposure, sensitivity, adaptive capacity).
    Used for prioritisation of adaptation investments and
    disclosure reporting.

    NEGLIGIBLE: Vulnerability score 0-20; no action required.
    LOW: Vulnerability score 20-40; monitor and review periodically.
    MODERATE: Vulnerability score 40-60; adaptation planning recommended.
    HIGH: Vulnerability score 60-80; adaptation investment required.
    CRITICAL: Vulnerability score 80-100; immediate adaptation essential.
    """

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# SDK Data Models (14)
# =============================================================================


class Location(BaseModel):
    """A geographic point location with optional metadata.

    Represents a WGS84 coordinate pair used to pin hazard data,
    assets, and risk indices to specific points on the Earth's
    surface. Coordinates follow the GeoJSON convention (longitude
    is X, latitude is Y).

    Attributes:
        latitude: WGS84 latitude in decimal degrees (-90 to 90).
        longitude: WGS84 longitude in decimal degrees (-180 to 180).
        elevation_m: Elevation above mean sea level in metres (optional).
        name: Human-readable place name or label.
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'US', 'DE').
    """

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees (-90 to 90)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees (-180 to 180)",
    )
    elevation_m: Optional[float] = Field(
        None,
        description="Elevation above mean sea level in metres",
    )
    name: str = Field(
        default="",
        description="Human-readable place name or label",
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code (e.g., 'US', 'DE')",
    )

    model_config = {"extra": "forbid"}

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within WGS84 bounds."""
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"latitude must be between -90 and 90, got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within WGS84 bounds."""
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"longitude must be between -180 and 180, got {v}")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country_code is empty or exactly 2 uppercase letters."""
        if v and (len(v) != 2 or not v.isalpha() or not v.isupper()):
            raise ValueError(
                f"country_code must be a 2-letter ISO 3166-1 alpha-2 code, got '{v}'"
            )
        return v


class HazardSource(BaseModel):
    """A registered external data source providing climate hazard information.

    Tracks the provenance and configuration of each data feed ingested
    into the Climate Hazard Connector. Sources are versioned and carry
    metadata for data quality assessment and audit trails.

    Attributes:
        source_id: Unique source identifier (UUID v4).
        name: Human-readable data source name.
        source_type: Classification of the data source methodology.
        hazard_types: List of hazard types provided by this source.
        coverage: Geographic or administrative coverage description.
        resolution: Spatial resolution description (e.g., '1km', '0.25deg').
        temporal_range: Temporal coverage description (e.g., '1979-2023').
        update_frequency: How often the source is updated (e.g., 'monthly').
        license: Data licence or usage terms identifier.
        url: URL or endpoint for accessing the data source.
        config: Source-specific connection and parsing configuration.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the source record for audit trail.
        registered_by: Actor that registered this source.
        registered_at: UTC timestamp when the source was registered.
        updated_at: UTC timestamp when the source was last modified.
    """

    source_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique source identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable data source name",
    )
    source_type: DataSourceType = Field(
        ...,
        description="Classification of the data source methodology",
    )
    hazard_types: List[HazardType] = Field(
        default_factory=list,
        description="List of hazard types provided by this source",
    )
    coverage: str = Field(
        default="global",
        description="Geographic or administrative coverage description",
    )
    resolution: str = Field(
        default="",
        description="Spatial resolution description (e.g., '1km', '0.25deg')",
    )
    temporal_range: str = Field(
        default="",
        description="Temporal coverage description (e.g., '1979-2023')",
    )
    update_frequency: str = Field(
        default="",
        description="How often the source is updated (e.g., 'monthly', 'annual')",
    )
    license: str = Field(
        default="",
        description="Data licence or usage terms identifier",
    )
    url: str = Field(
        default="",
        description="URL or endpoint for accessing the data source",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific connection and parsing configuration",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the source record for audit trail",
    )
    registered_by: str = Field(
        default="system",
        description="Actor that registered this source",
    )
    registered_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the source was registered",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the source was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: DataSourceType) -> DataSourceType:
        """Validate source_type is a valid enum member."""
        if not isinstance(v, DataSourceType):
            raise ValueError(f"Invalid source_type: {v}")
        return v


class HazardDataRecord(BaseModel):
    """A single hazard data observation or measurement from an external source.

    Represents one spatio-temporal data point ingested from a registered
    hazard data source. Records are immutable after ingestion and carry
    SHA-256 provenance hashes for tamper-evident audit trails.

    Attributes:
        record_id: Unique record identifier (UUID v4).
        source_id: ID of the registered data source that produced this record.
        hazard_type: Type of climate hazard this record describes.
        location: Geographic location of the observation.
        intensity: Hazard intensity value in source-native units.
        intensity_unit: Unit of measurement for the intensity value.
        probability: Annual exceedance probability (0.0 to 1.0) if available.
        frequency: Expected frequency of occurrence (events per year).
        duration_days: Expected or observed duration in days.
        return_period_years: Statistical return period in years.
        scenario: Climate scenario for projected data (None for historical).
        time_horizon: Projection time horizon (None for historical data).
        observed_at: UTC timestamp of the observation or model output date.
        metadata: Additional source-specific metadata.
        provenance_hash: SHA-256 hash of the record for audit trail.
        ingested_at: UTC timestamp when the record was ingested.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier (UUID v4)",
    )
    source_id: str = Field(
        ...,
        description="ID of the registered data source that produced this record",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard this record describes",
    )
    location: Location = Field(
        ...,
        description="Geographic location of the observation",
    )
    intensity: float = Field(
        default=0.0,
        ge=0.0,
        description="Hazard intensity value in source-native units",
    )
    intensity_unit: str = Field(
        default="",
        description="Unit of measurement for the intensity value (e.g., 'mm', 'C')",
    )
    probability: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Annual exceedance probability (0.0 to 1.0) if available",
    )
    frequency: Optional[float] = Field(
        None,
        ge=0.0,
        description="Expected frequency of occurrence (events per year)",
    )
    duration_days: Optional[float] = Field(
        None,
        ge=0.0,
        description="Expected or observed duration in days",
    )
    return_period_years: Optional[float] = Field(
        None,
        ge=0.0,
        description="Statistical return period in years (e.g., 100 for 1-in-100-year)",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario for projected data (None for historical)",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon (None for historical data)",
    )
    observed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the observation or model output date",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source-specific metadata",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the record for audit trail",
    )
    ingested_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the record was ingested",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, v: Optional[float]) -> Optional[float]:
        """Validate probability is in range [0.0, 1.0] if set."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"probability must be between 0.0 and 1.0, got {v}"
            )
        return v


class HazardEvent(BaseModel):
    """A recorded historical climate hazard event.

    Captures key attributes of a past hazard event from an event
    catalogue (e.g., EM-DAT, DesInventar). Events provide empirical
    evidence for calibrating risk models and validating projections.

    Attributes:
        event_id: Unique event identifier (UUID v4).
        hazard_type: Type of climate hazard that occurred.
        location: Geographic location or centroid of the event.
        start_date: UTC date when the event began.
        end_date: UTC date when the event ended (None if ongoing).
        intensity: Peak hazard intensity observed during the event.
        intensity_unit: Unit of measurement for the intensity value.
        affected_area_km2: Approximate area affected in square kilometres.
        deaths: Number of fatalities attributed to the event.
        injuries: Number of injuries attributed to the event.
        displaced: Number of people displaced by the event.
        economic_loss_usd: Estimated economic loss in US dollars.
        insured_loss_usd: Estimated insured loss in US dollars.
        source: Name or ID of the event catalogue source.
        source_event_id: Original event identifier in the source catalogue.
        description: Narrative description of the event.
        metadata: Additional event-specific metadata.
        provenance_hash: SHA-256 hash of the event record for audit trail.
        recorded_at: UTC timestamp when the event was recorded.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier (UUID v4)",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard that occurred",
    )
    location: Location = Field(
        ...,
        description="Geographic location or centroid of the event",
    )
    start_date: datetime = Field(
        ...,
        description="UTC date when the event began",
    )
    end_date: Optional[datetime] = Field(
        None,
        description="UTC date when the event ended (None if ongoing)",
    )
    intensity: float = Field(
        default=0.0,
        ge=0.0,
        description="Peak hazard intensity observed during the event",
    )
    intensity_unit: str = Field(
        default="",
        description="Unit of measurement for the intensity value",
    )
    affected_area_km2: Optional[float] = Field(
        None,
        ge=0.0,
        description="Approximate area affected in square kilometres",
    )
    deaths: int = Field(
        default=0,
        ge=0,
        description="Number of fatalities attributed to the event",
    )
    injuries: int = Field(
        default=0,
        ge=0,
        description="Number of injuries attributed to the event",
    )
    displaced: int = Field(
        default=0,
        ge=0,
        description="Number of people displaced by the event",
    )
    economic_loss_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated economic loss in US dollars",
    )
    insured_loss_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated insured loss in US dollars",
    )
    source: str = Field(
        default="",
        description="Name or ID of the event catalogue source",
    )
    source_event_id: str = Field(
        default="",
        description="Original event identifier in the source catalogue",
    )
    description: str = Field(
        default="",
        description="Narrative description of the event",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific metadata",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the event record for audit trail",
    )
    recorded_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the event was recorded",
    )

    model_config = {"extra": "forbid"}

    @field_validator("hazard_type")
    @classmethod
    def validate_hazard_type(cls, v: HazardType) -> HazardType:
        """Validate hazard_type is a valid enum member."""
        if not isinstance(v, HazardType):
            raise ValueError(f"Invalid hazard_type: {v}")
        return v

    @field_validator("deaths")
    @classmethod
    def validate_deaths(cls, v: int) -> int:
        """Validate deaths is non-negative."""
        if v < 0:
            raise ValueError(f"deaths must be >= 0, got {v}")
        return v


class RiskIndex(BaseModel):
    """A computed risk index for a specific hazard at a specific location.

    Represents the output of the risk calculation engine, combining
    probability, intensity, frequency, and duration into a composite
    risk score (0-100) with a qualitative risk level classification.

    Attributes:
        index_id: Unique risk index identifier (UUID v4).
        hazard_type: Type of climate hazard assessed.
        location: Geographic location of the risk assessment.
        risk_score: Composite risk score (0.0 to 100.0).
        risk_level: Qualitative risk classification derived from score.
        probability: Probability component of the risk score (0.0 to 1.0).
        intensity: Normalised intensity component (0.0 to 1.0).
        frequency: Normalised frequency component (0.0 to 1.0).
        duration: Normalised duration component (0.0 to 1.0).
        confidence: Confidence level of the risk score (0.0 to 1.0).
        scenario: Climate scenario used for the assessment.
        time_horizon: Projection time horizon used for the assessment.
        source_ids: IDs of data sources used in the calculation.
        methodology: Description of the calculation methodology applied.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the risk index for audit trail.
        calculated_by: Actor or service that calculated the risk index.
        calculated_at: UTC timestamp when the risk index was calculated.
    """

    index_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique risk index identifier (UUID v4)",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard assessed",
    )
    location: Location = Field(
        ...,
        description="Geographic location of the risk assessment",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite risk score (0.0 to 100.0)",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.NEGLIGIBLE,
        description="Qualitative risk classification derived from score",
    )
    probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability component of the risk score (0.0 to 1.0)",
    )
    intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised intensity component (0.0 to 1.0)",
    )
    frequency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised frequency component (0.0 to 1.0)",
    )
    duration: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised duration component (0.0 to 1.0)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the risk score (0.0 to 1.0)",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario used for the assessment",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon used for the assessment",
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="IDs of data sources used in the calculation",
    )
    methodology: str = Field(
        default="",
        description="Description of the calculation methodology applied",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the risk index for audit trail",
    )
    calculated_by: str = Field(
        default="system",
        description="Actor or service that calculated the risk index",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the risk index was calculated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("risk_score")
    @classmethod
    def validate_risk_score(cls, v: float) -> float:
        """Validate risk_score is in range [0.0, 100.0]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(
                f"risk_score must be between 0.0 and 100.0, got {v}"
            )
        return v

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"probability must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {v}"
            )
        return v


class ScenarioProjection(BaseModel):
    """A forward-looking risk projection under a specific climate scenario.

    Represents the output of the scenario projection engine, comparing
    baseline risk with projected risk under a specified SSP/RCP pathway
    and time horizon. Includes warming delta and scaling factor for
    transparency and reproducibility.

    Attributes:
        projection_id: Unique projection identifier (UUID v4).
        hazard_type: Type of climate hazard projected.
        location: Geographic location of the projection.
        scenario: IPCC climate scenario pathway.
        time_horizon: Projection time horizon.
        baseline_risk: Baseline risk score (0.0 to 100.0) from historical data.
        projected_risk: Projected risk score (0.0 to 100.0) under the scenario.
        risk_delta: Absolute change in risk score (projected - baseline).
        risk_delta_pct: Percentage change in risk score from baseline.
        warming_delta_c: Global mean temperature change in degrees Celsius.
        scaling_factor: Multiplicative scaling factor applied to baseline risk.
        confidence: Confidence level of the projection (0.0 to 1.0).
        model_ensemble: List of climate model names in the ensemble.
        methodology: Description of the projection methodology applied.
        provenance_hash: SHA-256 hash of the projection for audit trail.
        projected_by: Actor or service that calculated the projection.
        projected_at: UTC timestamp when the projection was calculated.
    """

    projection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique projection identifier (UUID v4)",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard projected",
    )
    location: Location = Field(
        ...,
        description="Geographic location of the projection",
    )
    scenario: Scenario = Field(
        ...,
        description="IPCC climate scenario pathway",
    )
    time_horizon: TimeHorizon = Field(
        ...,
        description="Projection time horizon",
    )
    baseline_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Baseline risk score (0.0 to 100.0) from historical data",
    )
    projected_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Projected risk score (0.0 to 100.0) under the scenario",
    )
    risk_delta: float = Field(
        default=0.0,
        description="Absolute change in risk score (projected minus baseline)",
    )
    risk_delta_pct: float = Field(
        default=0.0,
        description="Percentage change in risk score from baseline",
    )
    warming_delta_c: float = Field(
        default=0.0,
        ge=0.0,
        description="Global mean temperature change in degrees Celsius",
    )
    scaling_factor: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplicative scaling factor applied to baseline risk",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the projection (0.0 to 1.0)",
    )
    model_ensemble: List[str] = Field(
        default_factory=list,
        description="List of climate model names in the ensemble",
    )
    methodology: str = Field(
        default="",
        description="Description of the projection methodology applied",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the projection for audit trail",
    )
    projected_by: str = Field(
        default="system",
        description="Actor or service that calculated the projection",
    )
    projected_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the projection was calculated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("baseline_risk")
    @classmethod
    def validate_baseline_risk(cls, v: float) -> float:
        """Validate baseline_risk is in range [0.0, 100.0]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(
                f"baseline_risk must be between 0.0 and 100.0, got {v}"
            )
        return v

    @field_validator("projected_risk")
    @classmethod
    def validate_projected_risk(cls, v: float) -> float:
        """Validate projected_risk is in range [0.0, 100.0]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(
                f"projected_risk must be between 0.0 and 100.0, got {v}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {v}"
            )
        return v


class Asset(BaseModel):
    """A physical or operational asset registered for climate risk assessment.

    Represents any entity with a geographic location that is subject
    to climate-related physical risks. Assets are the primary targets
    for exposure analysis, vulnerability scoring, and adaptation
    screening within the Climate Hazard Connector.

    Attributes:
        asset_id: Unique asset identifier (UUID v4).
        name: Human-readable asset name or label.
        asset_type: Classification of the asset type.
        location: Geographic location of the asset.
        sector: Industry sector or classification (e.g., 'manufacturing').
        sub_sector: Sub-sector for finer classification.
        value_usd: Estimated asset value in US dollars.
        replacement_cost_usd: Estimated replacement cost in US dollars.
        operational_importance: Operational importance score (0.0 to 1.0).
        owner: Asset owner name or identifier.
        tags: Arbitrary key-value labels for filtering and grouping.
        metadata: Additional asset-specific metadata.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the asset record for audit trail.
        registered_by: Actor that registered this asset.
        registered_at: UTC timestamp when the asset was registered.
        updated_at: UTC timestamp when the asset was last modified.
    """

    asset_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique asset identifier (UUID v4)",
    )
    name: str = Field(
        ...,
        description="Human-readable asset name or label",
    )
    asset_type: AssetType = Field(
        ...,
        description="Classification of the asset type",
    )
    location: Location = Field(
        ...,
        description="Geographic location of the asset",
    )
    sector: str = Field(
        default="",
        description="Industry sector or classification (e.g., 'manufacturing')",
    )
    sub_sector: str = Field(
        default="",
        description="Sub-sector for finer classification (e.g., 'chemicals')",
    )
    value_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated asset value in US dollars",
    )
    replacement_cost_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated replacement cost in US dollars",
    )
    operational_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Operational importance score (0.0 to 1.0)",
    )
    owner: str = Field(
        default="",
        description="Asset owner name or identifier",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and grouping",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional asset-specific metadata",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the asset record for audit trail",
    )
    registered_by: str = Field(
        default="system",
        description="Actor that registered this asset",
    )
    registered_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the asset was registered",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the asset was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("asset_type")
    @classmethod
    def validate_asset_type(cls, v: AssetType) -> AssetType:
        """Validate asset_type is a valid enum member."""
        if not isinstance(v, AssetType):
            raise ValueError(f"Invalid asset_type: {v}")
        return v

    @field_validator("operational_importance")
    @classmethod
    def validate_operational_importance(cls, v: float) -> float:
        """Validate operational_importance is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"operational_importance must be between 0.0 and 1.0, got {v}"
            )
        return v


class ExposureResult(BaseModel):
    """Result of an exposure assessment for an asset against a specific hazard.

    Captures how exposed a registered asset is to a particular climate
    hazard, combining proximity, intensity at location, and frequency
    of exposure into a composite exposure score.

    Attributes:
        assessment_id: Unique assessment identifier (UUID v4).
        asset_id: ID of the assessed asset.
        hazard_type: Type of climate hazard assessed.
        exposure_level: Qualitative exposure classification.
        proximity_score: Proximity-based score (0.0 to 1.0; 1.0 = closest).
        intensity_at_location: Normalised hazard intensity at the asset (0-1).
        frequency_exposure: Normalised frequency of exposure (0.0 to 1.0).
        duration_exposure: Normalised expected duration of exposure (0.0 to 1.0).
        composite_score: Weighted composite exposure score (0.0 to 100.0).
        search_radius_km: Search radius used for proximity analysis (km).
        data_sources_used: Number of data sources used in the assessment.
        scenario: Climate scenario used for the assessment (optional).
        time_horizon: Projection time horizon used (optional).
        methodology: Description of the exposure assessment methodology.
        provenance_hash: SHA-256 hash of the assessment for audit trail.
        assessed_by: Actor or service that performed the assessment.
        assessed_at: UTC timestamp when the assessment was performed.
    """

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique assessment identifier (UUID v4)",
    )
    asset_id: str = Field(
        ...,
        description="ID of the assessed asset",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard assessed",
    )
    exposure_level: ExposureLevel = Field(
        default=ExposureLevel.NONE,
        description="Qualitative exposure classification",
    )
    proximity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proximity-based score (0.0 to 1.0; 1.0 = closest exposure)",
    )
    intensity_at_location: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised hazard intensity at the asset location (0.0 to 1.0)",
    )
    frequency_exposure: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised frequency of exposure (0.0 to 1.0)",
    )
    duration_exposure: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised expected duration of exposure (0.0 to 1.0)",
    )
    composite_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Weighted composite exposure score (0.0 to 100.0)",
    )
    search_radius_km: float = Field(
        default=50.0,
        ge=0.0,
        le=MAX_SEARCH_RADIUS_KM,
        description="Search radius used for proximity analysis (km)",
    )
    data_sources_used: int = Field(
        default=0,
        ge=0,
        description="Number of data sources used in the assessment",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario used for the assessment (optional)",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon used for the assessment (optional)",
    )
    methodology: str = Field(
        default="",
        description="Description of the exposure assessment methodology",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the assessment for audit trail",
    )
    assessed_by: str = Field(
        default="system",
        description="Actor or service that performed the assessment",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the assessment was performed",
    )

    model_config = {"extra": "forbid"}

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("asset_id must be non-empty")
        return v

    @field_validator("composite_score")
    @classmethod
    def validate_composite_score(cls, v: float) -> float:
        """Validate composite_score is in range [0.0, 100.0]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(
                f"composite_score must be between 0.0 and 100.0, got {v}"
            )
        return v

    @field_validator("proximity_score")
    @classmethod
    def validate_proximity_score(cls, v: float) -> float:
        """Validate proximity_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"proximity_score must be between 0.0 and 1.0, got {v}"
            )
        return v


class SensitivityProfile(BaseModel):
    """A sensitivity profile for an entity describing how strongly it is
    affected by climate hazards.

    Captures the factors that determine an asset's or organisation's
    sensitivity to physical climate risks. Factors are domain-specific
    (e.g., building material, crop type, elevation, water dependency)
    and combined into an overall sensitivity level.

    Attributes:
        profile_id: Unique profile identifier (UUID v4).
        entity_id: ID of the entity (asset, organisation, sector) profiled.
        entity_type: Type of entity profiled (e.g., 'asset', 'portfolio').
        sector: Industry sector for sector-specific sensitivity factors.
        factors: Dictionary of sensitivity factor names to scores (0.0 to 1.0).
        factor_weights: Optional weights for each factor (defaults to equal).
        overall_score: Weighted aggregate sensitivity score (0.0 to 1.0).
        overall_sensitivity: Qualitative sensitivity classification.
        hazard_sensitivities: Hazard-specific sensitivity overrides.
        description: Narrative description of the sensitivity profile.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the profile for audit trail.
        created_by: Actor that created the profile.
        created_at: UTC timestamp when the profile was created.
        updated_at: UTC timestamp when the profile was last modified.
    """

    profile_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique profile identifier (UUID v4)",
    )
    entity_id: str = Field(
        ...,
        description="ID of the entity (asset, organisation, sector) profiled",
    )
    entity_type: str = Field(
        default="asset",
        description="Type of entity profiled (e.g., 'asset', 'portfolio', 'sector')",
    )
    sector: str = Field(
        default="",
        description="Industry sector for sector-specific sensitivity factors",
    )
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of sensitivity factor names to scores (0.0 to 1.0)",
    )
    factor_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional weights for each factor (defaults to equal weighting)",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted aggregate sensitivity score (0.0 to 1.0)",
    )
    overall_sensitivity: SensitivityLevel = Field(
        default=SensitivityLevel.MODERATE,
        description="Qualitative sensitivity classification",
    )
    hazard_sensitivities: Dict[str, float] = Field(
        default_factory=dict,
        description="Hazard-specific sensitivity overrides (hazard_type -> score)",
    )
    description: str = Field(
        default="",
        description="Narrative description of the sensitivity profile",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the profile for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor that created the profile",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the profile was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the profile was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        """Validate overall_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"overall_score must be between 0.0 and 1.0, got {v}"
            )
        return v


class AdaptiveCapacityProfile(BaseModel):
    """An adaptive capacity profile for an entity describing its ability
    to cope with or adjust to climate hazards.

    Captures indicators that determine how well an asset, organisation,
    or system can adapt to physical climate risks. Indicators include
    financial reserves, redundancy, insurance coverage, contingency
    planning, and infrastructure resilience.

    Attributes:
        profile_id: Unique profile identifier (UUID v4).
        entity_id: ID of the entity profiled.
        entity_type: Type of entity profiled (e.g., 'asset', 'portfolio').
        indicators: Dictionary of capacity indicator names to scores (0.0 to 1.0).
        indicator_weights: Optional weights for each indicator.
        overall_score: Weighted aggregate adaptive capacity score (0.0 to 1.0).
        overall_capacity: Qualitative adaptive capacity classification.
        financial_reserves_score: Financial resilience sub-score (0.0 to 1.0).
        redundancy_score: Operational redundancy sub-score (0.0 to 1.0).
        insurance_score: Insurance coverage sub-score (0.0 to 1.0).
        contingency_score: Contingency planning sub-score (0.0 to 1.0).
        infrastructure_score: Infrastructure resilience sub-score (0.0 to 1.0).
        description: Narrative description of the adaptive capacity profile.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the profile for audit trail.
        created_by: Actor that created the profile.
        created_at: UTC timestamp when the profile was created.
        updated_at: UTC timestamp when the profile was last modified.
    """

    profile_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique profile identifier (UUID v4)",
    )
    entity_id: str = Field(
        ...,
        description="ID of the entity profiled",
    )
    entity_type: str = Field(
        default="asset",
        description="Type of entity profiled (e.g., 'asset', 'portfolio', 'sector')",
    )
    indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of capacity indicator names to scores (0.0 to 1.0)",
    )
    indicator_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional weights for each indicator (defaults to equal weighting)",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted aggregate adaptive capacity score (0.0 to 1.0)",
    )
    overall_capacity: AdaptiveCapacity = Field(
        default=AdaptiveCapacity.MODERATE,
        description="Qualitative adaptive capacity classification",
    )
    financial_reserves_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Financial resilience sub-score (0.0 to 1.0)",
    )
    redundancy_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Operational redundancy sub-score (0.0 to 1.0)",
    )
    insurance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Insurance coverage sub-score (0.0 to 1.0)",
    )
    contingency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Contingency planning sub-score (0.0 to 1.0)",
    )
    infrastructure_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Infrastructure resilience sub-score (0.0 to 1.0)",
    )
    description: str = Field(
        default="",
        description="Narrative description of the adaptive capacity profile",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the profile for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor that created the profile",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the profile was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the profile was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @field_validator("overall_score")
    @classmethod
    def validate_overall_score(cls, v: float) -> float:
        """Validate overall_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"overall_score must be between 0.0 and 1.0, got {v}"
            )
        return v


class VulnerabilityScore(BaseModel):
    """A composite vulnerability score combining exposure, sensitivity, and
    adaptive capacity.

    Follows the IPCC AR5/AR6 vulnerability framework:
        vulnerability = f(exposure, sensitivity, adaptive capacity)
    where higher exposure and sensitivity increase vulnerability,
    and higher adaptive capacity reduces it. The composite score
    is deterministic and reproducible via SHA-256 provenance hashing.

    Attributes:
        score_id: Unique vulnerability score identifier (UUID v4).
        entity_id: ID of the entity (asset, portfolio, etc.) scored.
        hazard_type: Type of climate hazard assessed.
        exposure_score: Normalised exposure score (0.0 to 1.0).
        sensitivity_score: Normalised sensitivity score (0.0 to 1.0).
        adaptive_capacity_score: Normalised adaptive capacity score (0.0 to 1.0).
        vulnerability_score: Composite vulnerability score (0.0 to 100.0).
        vulnerability_level: Qualitative vulnerability classification.
        exposure_weight: Weight assigned to exposure in composite (0.0 to 1.0).
        sensitivity_weight: Weight assigned to sensitivity (0.0 to 1.0).
        capacity_weight: Weight assigned to adaptive capacity (0.0 to 1.0).
        confidence: Confidence level of the vulnerability score (0.0 to 1.0).
        scenario: Climate scenario used for the assessment (optional).
        time_horizon: Projection time horizon used (optional).
        methodology: Description of the vulnerability scoring methodology.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the score for audit trail.
        scored_by: Actor or service that calculated the score.
        scored_at: UTC timestamp when the score was calculated.
    """

    score_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique vulnerability score identifier (UUID v4)",
    )
    entity_id: str = Field(
        ...,
        description="ID of the entity (asset, portfolio, etc.) scored",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard assessed",
    )
    exposure_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised exposure score (0.0 to 1.0)",
    )
    sensitivity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised sensitivity score (0.0 to 1.0)",
    )
    adaptive_capacity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised adaptive capacity score (0.0 to 1.0)",
    )
    vulnerability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite vulnerability score (0.0 to 100.0)",
    )
    vulnerability_level: VulnerabilityLevel = Field(
        default=VulnerabilityLevel.NEGLIGIBLE,
        description="Qualitative vulnerability classification",
    )
    exposure_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight assigned to exposure in composite (0.0 to 1.0)",
    )
    sensitivity_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight assigned to sensitivity in composite (0.0 to 1.0)",
    )
    capacity_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight assigned to adaptive capacity in composite (0.0 to 1.0)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the vulnerability score (0.0 to 1.0)",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario used for the assessment (optional)",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon used for the assessment (optional)",
    )
    methodology: str = Field(
        default="ipcc_ar6",
        description="Description of the vulnerability scoring methodology",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the score for audit trail",
    )
    scored_by: str = Field(
        default="system",
        description="Actor or service that calculated the score",
    )
    scored_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the score was calculated",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @field_validator("vulnerability_score")
    @classmethod
    def validate_vulnerability_score(cls, v: float) -> float:
        """Validate vulnerability_score is in range [0.0, 100.0]."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(
                f"vulnerability_score must be between 0.0 and 100.0, got {v}"
            )
        return v

    @field_validator("exposure_score")
    @classmethod
    def validate_exposure_score(cls, v: float) -> float:
        """Validate exposure_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"exposure_score must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("sensitivity_score")
    @classmethod
    def validate_sensitivity_score(cls, v: float) -> float:
        """Validate sensitivity_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"sensitivity_score must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("adaptive_capacity_score")
    @classmethod
    def validate_adaptive_capacity_score(cls, v: float) -> float:
        """Validate adaptive_capacity_score is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"adaptive_capacity_score must be between 0.0 and 1.0, got {v}"
            )
        return v


class CompoundHazard(BaseModel):
    """A compound (multi-hazard) interaction definition.

    Describes how two or more climate hazards interact to produce
    amplified or correlated risks. For example, extreme precipitation
    combined with deforestation amplifies landslide risk. Compound
    hazards are used by the risk engine to adjust single-hazard
    risk indices when multiple hazards are present.

    Attributes:
        compound_id: Unique compound hazard identifier (UUID v4).
        name: Human-readable name for this compound hazard.
        primary_hazard: The primary (triggering) hazard type.
        secondary_hazards: List of secondary (interacting) hazard types.
        correlation_factor: Pearson correlation between primary and secondaries.
        amplification_factor: Multiplicative amplification of combined risk.
        interaction_type: Nature of the interaction (e.g., 'cascading', 'concurrent').
        description: Narrative description of the compound hazard mechanism.
        evidence_sources: References or citations supporting the interaction.
        applicable_regions: Geographic regions where this interaction is documented.
        applicable_scenarios: Climate scenarios where interaction is significant.
        confidence: Confidence level of the compound hazard estimate (0.0 to 1.0).
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the compound hazard for audit trail.
        created_by: Actor that created the compound hazard definition.
        created_at: UTC timestamp when the definition was created.
        updated_at: UTC timestamp when the definition was last modified.
    """

    compound_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique compound hazard identifier (UUID v4)",
    )
    name: str = Field(
        default="",
        description="Human-readable name for this compound hazard",
    )
    primary_hazard: HazardType = Field(
        ...,
        description="The primary (triggering) hazard type",
    )
    secondary_hazards: List[HazardType] = Field(
        default_factory=list,
        description="List of secondary (interacting) hazard types",
    )
    correlation_factor: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Pearson correlation between primary and secondaries (-1 to 1)",
    )
    amplification_factor: float = Field(
        default=1.0,
        ge=0.0,
        description="Multiplicative amplification of combined risk (>= 0.0)",
    )
    interaction_type: str = Field(
        default="concurrent",
        description="Nature of the interaction (e.g., 'cascading', 'concurrent', 'compounding')",
    )
    description: str = Field(
        default="",
        description="Narrative description of the compound hazard mechanism",
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="References or citations supporting the interaction",
    )
    applicable_regions: List[str] = Field(
        default_factory=list,
        description="Geographic regions where this interaction is documented",
    )
    applicable_scenarios: List[Scenario] = Field(
        default_factory=list,
        description="Climate scenarios where interaction is significant",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the compound hazard estimate (0.0 to 1.0)",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the compound hazard for audit trail",
    )
    created_by: str = Field(
        default="system",
        description="Actor that created the compound hazard definition",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the definition was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the definition was last modified",
    )

    model_config = {"extra": "forbid"}

    @field_validator("primary_hazard")
    @classmethod
    def validate_primary_hazard(cls, v: HazardType) -> HazardType:
        """Validate primary_hazard is a valid enum member."""
        if not isinstance(v, HazardType):
            raise ValueError(f"Invalid primary_hazard: {v}")
        return v

    @field_validator("correlation_factor")
    @classmethod
    def validate_correlation_factor(cls, v: float) -> float:
        """Validate correlation_factor is in range [-1.0, 1.0]."""
        if not (-1.0 <= v <= 1.0):
            raise ValueError(
                f"correlation_factor must be between -1.0 and 1.0, got {v}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {v}"
            )
        return v


class ComplianceReport(BaseModel):
    """A generated climate risk compliance report in a specified format.

    Produced by the reporting engine to render climate risk assessment
    results for regulatory disclosure (TCFD, CSRD/ESRS, SEC Climate,
    IFRS S2) or internal decision-making. Reports are immutable once
    generated and include a SHA-256 hash for tamper detection.

    Attributes:
        report_id: Unique report identifier (UUID v4).
        report_type: Type of climate risk report generated.
        format: Output format of the report.
        framework: Compliance framework alignment (e.g., 'tcfd', 'csrd_esrs').
        title: Human-readable report title.
        description: Brief description of the report scope and contents.
        scope: Scope of the report (e.g., 'portfolio', 'asset:xyz').
        parameters: Report generation parameters and configuration.
        content: The rendered report content as a string.
        report_hash: SHA-256 hash of the report content for tamper detection.
        asset_count: Number of assets covered in the report.
        hazard_count: Number of hazard types analysed in the report.
        scenario_count: Number of scenarios included in the report.
        time_horizons: Time horizons included in the report.
        risk_summary: Aggregate risk metrics for the report scope.
        recommendations: List of adaptation or mitigation recommendations.
        generated_by: Actor (user or service) that requested the report.
        generated_at: UTC timestamp when the report was generated.
        provenance_hash: SHA-256 hash of the full report record for audit trail.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier (UUID v4)",
    )
    report_type: ReportType = Field(
        default=ReportType.PHYSICAL_RISK_ASSESSMENT,
        description="Type of climate risk report generated",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format of the report",
    )
    framework: str = Field(
        default="tcfd",
        description="Compliance framework alignment (e.g., 'tcfd', 'csrd_esrs')",
    )
    title: str = Field(
        default="",
        description="Human-readable report title",
    )
    description: str = Field(
        default="",
        description="Brief description of the report scope and contents",
    )
    scope: str = Field(
        default="full",
        description="Scope of the report (e.g., 'portfolio', 'asset:xyz')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report generation parameters and configuration",
    )
    content: str = Field(
        default="",
        description="The rendered report content as a string",
    )
    report_hash: str = Field(
        default="",
        description="SHA-256 hash of the report content for tamper detection",
    )
    asset_count: int = Field(
        default=0,
        ge=0,
        description="Number of assets covered in the report",
    )
    hazard_count: int = Field(
        default=0,
        ge=0,
        description="Number of hazard types analysed in the report",
    )
    scenario_count: int = Field(
        default=0,
        ge=0,
        description="Number of scenarios included in the report",
    )
    time_horizons: List[TimeHorizon] = Field(
        default_factory=list,
        description="Time horizons included in the report",
    )
    risk_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate risk metrics for the report scope",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of adaptation or mitigation recommendations",
    )
    generated_by: str = Field(
        default="system",
        description="Actor (user or service) that requested the report",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the report was generated",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the full report record for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Validate framework is non-empty."""
        if not v or not v.strip():
            raise ValueError("framework must be non-empty")
        return v


class PipelineRun(BaseModel):
    """A record of a complete Climate Hazard Connector pipeline execution.

    Tracks the end-to-end execution of the climate hazard assessment
    pipeline, including stage completion status, aggregate results,
    performance metrics, and provenance hashing. Pipeline runs are
    immutable after completion.

    Attributes:
        pipeline_id: Unique pipeline run identifier (UUID v4).
        name: Human-readable name for this pipeline run.
        status: Current status of the pipeline run (e.g., 'running', 'completed').
        stages_completed: List of completed pipeline stage names.
        stages_total: Total number of pipeline stages.
        current_stage: Name of the currently executing stage.
        results: Aggregate results from all completed stages.
        errors: List of error messages from failed stages.
        warnings: List of warning messages from pipeline stages.
        assets_processed: Number of assets processed in this run.
        hazards_assessed: Number of hazard types assessed in this run.
        scenarios_projected: Number of scenario projections generated.
        vulnerabilities_scored: Number of vulnerability scores calculated.
        duration_ms: Total wall-clock duration in milliseconds.
        config: Pipeline configuration used for this run.
        namespace: Tenant or organizational namespace for isolation.
        provenance_hash: SHA-256 hash of the pipeline run for audit trail.
        triggered_by: Actor or service that triggered the pipeline.
        started_at: UTC timestamp when the pipeline started.
        completed_at: UTC timestamp when the pipeline completed (None if running).
    """

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique pipeline run identifier (UUID v4)",
    )
    name: str = Field(
        default="",
        description="Human-readable name for this pipeline run",
    )
    status: str = Field(
        default="pending",
        description="Current status (pending, running, completed, failed, cancelled)",
    )
    stages_completed: List[str] = Field(
        default_factory=list,
        description="List of completed pipeline stage names",
    )
    stages_total: int = Field(
        default=7,
        ge=0,
        description="Total number of pipeline stages",
    )
    current_stage: str = Field(
        default="",
        description="Name of the currently executing stage",
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate results from all completed stages",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages from failed stages",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages from pipeline stages",
    )
    assets_processed: int = Field(
        default=0,
        ge=0,
        description="Number of assets processed in this run",
    )
    hazards_assessed: int = Field(
        default=0,
        ge=0,
        description="Number of hazard types assessed in this run",
    )
    scenarios_projected: int = Field(
        default=0,
        ge=0,
        description="Number of scenario projections generated",
    )
    vulnerabilities_scored: int = Field(
        default=0,
        ge=0,
        description="Number of vulnerability scores calculated",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock duration in milliseconds",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline configuration used for this run",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the pipeline run for audit trail",
    )
    triggered_by: str = Field(
        default="system",
        description="Actor or service that triggered the pipeline",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the pipeline started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the pipeline completed (None if running)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of the allowed values."""
        allowed = {"pending", "running", "completed", "failed", "cancelled"}
        if v not in allowed:
            raise ValueError(
                f"status must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


# =============================================================================
# Request Models (8)
# =============================================================================


class RegisterSourceRequest(BaseModel):
    """Request body for registering a new hazard data source.

    Attributes:
        name: Human-readable data source name.
        source_type: Classification of the data source methodology.
        hazard_types: List of hazard types provided by this source.
        coverage: Geographic or administrative coverage description.
        resolution: Spatial resolution description.
        temporal_range: Temporal coverage description.
        update_frequency: How often the source is updated.
        license: Data licence or usage terms identifier.
        url: URL or endpoint for accessing the data source.
        config: Source-specific connection and parsing configuration.
        namespace: Tenant or organizational namespace for isolation.
    """

    name: str = Field(
        ...,
        description="Human-readable data source name",
    )
    source_type: DataSourceType = Field(
        ...,
        description="Classification of the data source methodology",
    )
    hazard_types: List[HazardType] = Field(
        default_factory=list,
        description="List of hazard types provided by this source",
    )
    coverage: str = Field(
        default="global",
        description="Geographic or administrative coverage description",
    )
    resolution: str = Field(
        default="",
        description="Spatial resolution description (e.g., '1km', '0.25deg')",
    )
    temporal_range: str = Field(
        default="",
        description="Temporal coverage description (e.g., '1979-2023')",
    )
    update_frequency: str = Field(
        default="",
        description="How often the source is updated (e.g., 'monthly', 'annual')",
    )
    license: str = Field(
        default="",
        description="Data licence or usage terms identifier",
    )
    url: str = Field(
        default="",
        description="URL or endpoint for accessing the data source",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific connection and parsing configuration",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class IngestDataRequest(BaseModel):
    """Request body for ingesting hazard data records from a registered source.

    Attributes:
        source_id: ID of the registered data source providing the records.
        records: List of hazard data records to ingest.
        batch_size: Number of records to process per ingestion chunk.
        validate_coordinates: Whether to validate WGS84 coordinate bounds.
        deduplicate: Whether to deduplicate records by location and timestamp.
        overwrite_existing: Whether to overwrite existing records for same key.
        namespace: Tenant or organizational namespace for isolation.
    """

    source_id: str = Field(
        ...,
        description="ID of the registered data source providing the records",
    )
    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of hazard data records to ingest (raw or structured)",
    )
    batch_size: int = Field(
        default=DEFAULT_PIPELINE_BATCH_SIZE,
        ge=1,
        le=MAX_RECORDS_PER_BATCH,
        description="Number of records to process per ingestion chunk",
    )
    validate_coordinates: bool = Field(
        default=True,
        description="Whether to validate WGS84 coordinate bounds on ingestion",
    )
    deduplicate: bool = Field(
        default=True,
        description="Whether to deduplicate records by location and timestamp",
    )
    overwrite_existing: bool = Field(
        default=False,
        description="Whether to overwrite existing records for the same key",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch_size is within allowed range."""
        if v < 1 or v > MAX_RECORDS_PER_BATCH:
            raise ValueError(
                f"batch_size must be between 1 and {MAX_RECORDS_PER_BATCH}, got {v}"
            )
        return v


class CalculateRiskRequest(BaseModel):
    """Request body for calculating a risk index for a hazard at a location.

    Attributes:
        hazard_type: Type of climate hazard to assess.
        location: Geographic location for the risk calculation.
        scenario: Climate scenario for forward-looking risk (optional).
        time_horizon: Projection time horizon (optional).
        source_ids: Optional list of specific source IDs to use.
        include_compound: Whether to include compound hazard adjustments.
        search_radius_km: Search radius for data aggregation (km).
        methodology: Methodology to use for risk calculation.
        namespace: Tenant or organizational namespace for isolation.
    """

    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard to assess",
    )
    location: Location = Field(
        ...,
        description="Geographic location for the risk calculation",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario for forward-looking risk (optional)",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon (optional)",
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="Optional list of specific source IDs to use (empty = all)",
    )
    include_compound: bool = Field(
        default=False,
        description="Whether to include compound hazard adjustments",
    )
    search_radius_km: float = Field(
        default=50.0,
        ge=0.0,
        le=MAX_SEARCH_RADIUS_KM,
        description="Search radius for data aggregation in kilometres",
    )
    methodology: str = Field(
        default="default",
        description="Methodology to use for risk calculation (e.g., 'default', 'weighted')",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("search_radius_km")
    @classmethod
    def validate_search_radius_km(cls, v: float) -> float:
        """Validate search_radius_km is within allowed range."""
        if v < 0.0 or v > MAX_SEARCH_RADIUS_KM:
            raise ValueError(
                f"search_radius_km must be between 0.0 and {MAX_SEARCH_RADIUS_KM}, got {v}"
            )
        return v


class ProjectScenarioRequest(BaseModel):
    """Request body for projecting risk under a specific climate scenario.

    Attributes:
        hazard_type: Type of climate hazard to project.
        location: Geographic location for the projection.
        scenario: IPCC climate scenario pathway to project under.
        time_horizon: Projection time horizon.
        baseline_source_ids: Optional source IDs for baseline calculation.
        warming_delta_c: Optional manual warming delta override.
        include_ensemble_spread: Whether to include model ensemble uncertainty.
        methodology: Methodology to use for projection.
        namespace: Tenant or organizational namespace for isolation.
    """

    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard to project",
    )
    location: Location = Field(
        ...,
        description="Geographic location for the projection",
    )
    scenario: Scenario = Field(
        ...,
        description="IPCC climate scenario pathway to project under",
    )
    time_horizon: TimeHorizon = Field(
        ...,
        description="Projection time horizon",
    )
    baseline_source_ids: List[str] = Field(
        default_factory=list,
        description="Optional source IDs for baseline risk calculation (empty = all)",
    )
    warming_delta_c: Optional[float] = Field(
        None,
        ge=0.0,
        description="Optional manual warming delta override in degrees Celsius",
    )
    include_ensemble_spread: bool = Field(
        default=False,
        description="Whether to include model ensemble uncertainty bounds",
    )
    methodology: str = Field(
        default="default",
        description="Methodology for projection (e.g., 'default', 'pattern_scaling')",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}


class RegisterAssetRequest(BaseModel):
    """Request body for registering a new asset for climate risk assessment.

    Attributes:
        name: Human-readable asset name or label.
        asset_type: Classification of the asset type.
        location: Geographic location of the asset.
        sector: Industry sector or classification.
        sub_sector: Sub-sector for finer classification.
        value_usd: Estimated asset value in US dollars.
        replacement_cost_usd: Estimated replacement cost in US dollars.
        operational_importance: Operational importance score (0.0 to 1.0).
        owner: Asset owner name or identifier.
        tags: Arbitrary key-value labels for filtering and grouping.
        metadata: Additional asset-specific metadata.
        namespace: Tenant or organizational namespace for isolation.
    """

    name: str = Field(
        ...,
        description="Human-readable asset name or label",
    )
    asset_type: AssetType = Field(
        ...,
        description="Classification of the asset type",
    )
    location: Location = Field(
        ...,
        description="Geographic location of the asset",
    )
    sector: str = Field(
        default="",
        description="Industry sector or classification (e.g., 'manufacturing')",
    )
    sub_sector: str = Field(
        default="",
        description="Sub-sector for finer classification (e.g., 'chemicals')",
    )
    value_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated asset value in US dollars",
    )
    replacement_cost_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated replacement cost in US dollars",
    )
    operational_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Operational importance score (0.0 to 1.0)",
    )
    owner: str = Field(
        default="",
        description="Asset owner name or identifier",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value labels for filtering and grouping",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional asset-specific metadata",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("operational_importance")
    @classmethod
    def validate_operational_importance(cls, v: float) -> float:
        """Validate operational_importance is in range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"operational_importance must be between 0.0 and 1.0, got {v}"
            )
        return v


class AssessExposureRequest(BaseModel):
    """Request body for assessing an asset's exposure to a specific hazard.

    Attributes:
        asset_id: ID of the asset to assess.
        hazard_type: Type of climate hazard to assess exposure for.
        scenario: Climate scenario for forward-looking exposure (optional).
        time_horizon: Projection time horizon (optional).
        search_radius_km: Search radius for proximity analysis (km).
        source_ids: Optional list of specific source IDs to use.
        include_compound: Whether to include compound hazard adjustments.
        methodology: Methodology to use for exposure assessment.
        namespace: Tenant or organizational namespace for isolation.
    """

    asset_id: str = Field(
        ...,
        description="ID of the asset to assess",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard to assess exposure for",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario for forward-looking exposure (optional)",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon (optional)",
    )
    search_radius_km: float = Field(
        default=50.0,
        ge=0.0,
        le=MAX_SEARCH_RADIUS_KM,
        description="Search radius for proximity analysis in kilometres",
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="Optional list of specific source IDs to use (empty = all)",
    )
    include_compound: bool = Field(
        default=False,
        description="Whether to include compound hazard adjustments",
    )
    methodology: str = Field(
        default="default",
        description="Methodology for exposure assessment (e.g., 'default', 'proximity')",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("asset_id")
    @classmethod
    def validate_asset_id(cls, v: str) -> str:
        """Validate asset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("asset_id must be non-empty")
        return v

    @field_validator("search_radius_km")
    @classmethod
    def validate_search_radius_km(cls, v: float) -> float:
        """Validate search_radius_km is within allowed range."""
        if v < 0.0 or v > MAX_SEARCH_RADIUS_KM:
            raise ValueError(
                f"search_radius_km must be between 0.0 and {MAX_SEARCH_RADIUS_KM}, got {v}"
            )
        return v


class ScoreVulnerabilityRequest(BaseModel):
    """Request body for scoring the vulnerability of an entity to a hazard.

    Attributes:
        entity_id: ID of the entity (asset, portfolio) to score.
        hazard_type: Type of climate hazard to assess vulnerability for.
        exposure_score: Pre-calculated exposure score (0.0 to 1.0) or None.
        sensitivity_profile_id: ID of a saved sensitivity profile (optional).
        adaptive_capacity_profile_id: ID of a saved capacity profile (optional).
        sensitivity_factors: Inline sensitivity factors (if no profile ID).
        adaptive_capacity_indicators: Inline capacity indicators (if no profile ID).
        exposure_weight: Weight for exposure in composite (0.0 to 1.0).
        sensitivity_weight: Weight for sensitivity in composite (0.0 to 1.0).
        capacity_weight: Weight for adaptive capacity in composite (0.0 to 1.0).
        scenario: Climate scenario for forward-looking vulnerability (optional).
        time_horizon: Projection time horizon (optional).
        methodology: Methodology for vulnerability scoring.
        namespace: Tenant or organizational namespace for isolation.
    """

    entity_id: str = Field(
        ...,
        description="ID of the entity (asset, portfolio) to score",
    )
    hazard_type: HazardType = Field(
        ...,
        description="Type of climate hazard to assess vulnerability for",
    )
    exposure_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Pre-calculated exposure score (0.0 to 1.0) or None for auto",
    )
    sensitivity_profile_id: Optional[str] = Field(
        None,
        description="ID of a saved sensitivity profile to use (optional)",
    )
    adaptive_capacity_profile_id: Optional[str] = Field(
        None,
        description="ID of a saved adaptive capacity profile to use (optional)",
    )
    sensitivity_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Inline sensitivity factors (used if no profile ID provided)",
    )
    adaptive_capacity_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Inline capacity indicators (used if no profile ID provided)",
    )
    exposure_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for exposure in composite (0.0 to 1.0)",
    )
    sensitivity_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for sensitivity in composite (0.0 to 1.0)",
    )
    capacity_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for adaptive capacity in composite (0.0 to 1.0)",
    )
    scenario: Optional[Scenario] = Field(
        None,
        description="Climate scenario for forward-looking vulnerability (optional)",
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None,
        description="Projection time horizon (optional)",
    )
    methodology: str = Field(
        default="ipcc_ar6",
        description="Methodology for vulnerability scoring (e.g., 'ipcc_ar6')",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v

    @field_validator("exposure_score")
    @classmethod
    def validate_exposure_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate exposure_score is in range [0.0, 1.0] if set."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"exposure_score must be between 0.0 and 1.0, got {v}"
            )
        return v


class GenerateReportRequest(BaseModel):
    """Request body for generating a climate risk compliance report.

    Triggers the reporting engine to produce a climate risk report in
    the specified format and type, scoped to the requested assets,
    hazards, scenarios, and time horizons.

    Attributes:
        report_type: Type of climate risk report to generate.
        report_format: Output format for the report.
        framework: Compliance framework alignment.
        title: Human-readable report title.
        scope: Scope of the report (e.g., 'portfolio', 'asset:xyz').
        asset_ids: Optional list of asset IDs to include in the report.
        hazard_types: Optional list of hazard types to include.
        scenarios: Optional list of scenarios to include.
        time_horizons: Optional list of time horizons to include.
        parameters: Report generation parameters and configuration.
        include_recommendations: Whether to include adaptation recommendations.
        include_maps: Whether to include geographic map visualisations.
        include_projections: Whether to include scenario projection charts.
        namespace: Tenant or organizational namespace for isolation.
    """

    report_type: ReportType = Field(
        default=ReportType.PHYSICAL_RISK_ASSESSMENT,
        description="Type of climate risk report to generate",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format for the report",
    )
    framework: str = Field(
        default="tcfd",
        description="Compliance framework alignment (e.g., 'tcfd', 'csrd_esrs')",
    )
    title: str = Field(
        default="",
        description="Human-readable report title",
    )
    scope: str = Field(
        default="full",
        description="Scope of the report (e.g., 'portfolio', 'asset:xyz', 'full')",
    )
    asset_ids: List[str] = Field(
        default_factory=list,
        description="Optional list of asset IDs to include (empty = all)",
    )
    hazard_types: List[HazardType] = Field(
        default_factory=list,
        description="Optional list of hazard types to include (empty = all)",
    )
    scenarios: List[Scenario] = Field(
        default_factory=list,
        description="Optional list of scenarios to include (empty = all)",
    )
    time_horizons: List[TimeHorizon] = Field(
        default_factory=list,
        description="Optional list of time horizons to include (empty = all)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report generation parameters and configuration",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to include adaptation recommendations in the report",
    )
    include_maps: bool = Field(
        default=False,
        description="Whether to include geographic map visualisations",
    )
    include_projections: bool = Field(
        default=True,
        description="Whether to include scenario projection charts",
    )
    namespace: str = Field(
        default="default",
        description="Tenant or organizational namespace for isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Validate framework is non-empty."""
        if not v or not v.strip():
            raise ValueError("framework must be non-empty")
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (gis_connector)
    # -------------------------------------------------------------------------
    "SpatialAnalyzerEngine",
    "BoundaryResolverEngine",
    "CRSTransformerEngine",
    # -------------------------------------------------------------------------
    # Availability flags (for downstream feature detection)
    # -------------------------------------------------------------------------
    "_SAE_AVAILABLE",
    "_BRE_AVAILABLE",
    "_CTE_AVAILABLE",
    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    "VERSION",
    "MAX_SOURCES_PER_NAMESPACE",
    "MAX_RECORDS_PER_BATCH",
    "MAX_ASSETS_PER_NAMESPACE",
    "MAX_COMPOUND_HAZARDS",
    "MAX_FACTORS_PER_PROFILE",
    "DEFAULT_PIPELINE_BATCH_SIZE",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MAX_PROJECTIONS_PER_PAIR",
    "MAX_SEARCH_RADIUS_KM",
    "RISK_LEVEL_BOUNDARIES",
    "SUPPORTED_SCENARIOS",
    "SUPPORTED_REPORT_FORMATS",
    "SUPPORTED_FRAMEWORKS",
    "TIME_HORIZON_RANGES",
    # -------------------------------------------------------------------------
    # Enumerations (12)
    # -------------------------------------------------------------------------
    "HazardType",
    "RiskLevel",
    "Scenario",
    "TimeHorizon",
    "AssetType",
    "ReportType",
    "ReportFormat",
    "DataSourceType",
    "ExposureLevel",
    "SensitivityLevel",
    "AdaptiveCapacity",
    "VulnerabilityLevel",
    # -------------------------------------------------------------------------
    # SDK data models (14)
    # -------------------------------------------------------------------------
    "Location",
    "HazardSource",
    "HazardDataRecord",
    "HazardEvent",
    "RiskIndex",
    "ScenarioProjection",
    "Asset",
    "ExposureResult",
    "SensitivityProfile",
    "AdaptiveCapacityProfile",
    "VulnerabilityScore",
    "CompoundHazard",
    "ComplianceReport",
    "PipelineRun",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "RegisterSourceRequest",
    "IngestDataRequest",
    "CalculateRiskRequest",
    "ProjectScenarioRequest",
    "RegisterAssetRequest",
    "AssessExposureRequest",
    "ScoreVulnerabilityRequest",
    "GenerateReportRequest",
]
