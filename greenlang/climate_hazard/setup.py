# -*- coding: utf-8 -*-
"""
Climate Hazard Connector Service Setup - AGENT-DATA-020

Provides ``configure_climate_hazard(app)`` which wires up the Climate
Hazard Connector Agent SDK (hazard database, risk index, scenario
projector, exposure assessor, vulnerability scorer, compliance reporter,
hazard pipeline, provenance tracker) and mounts the REST API.

Also exposes ``get_service()`` for programmatic access and the
``ClimateHazardService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.climate_hazard.setup import configure_climate_hazard
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_climate_hazard(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.climate_hazard.config import (
    ClimateHazardConfig,
    get_config,
)
from greenlang.climate_hazard.metrics import (
    PROMETHEUS_AVAILABLE,
    record_ingestion,
    record_risk_calculation,
    record_projection,
    record_exposure,
    record_vulnerability,
    record_report,
    record_pipeline,
    set_active_sources,
    set_active_assets,
    set_high_risk,
    observe_ingestion_duration,
    observe_pipeline_duration,
)
from greenlang.climate_hazard.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.hazard_database import HazardDatabaseEngine
except ImportError:
    HazardDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.risk_index import RiskIndexEngine
except ImportError:
    RiskIndexEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.scenario_projector import ScenarioProjectorEngine
except ImportError:
    ScenarioProjectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.exposure_assessor import ExposureAssessorEngine
except ImportError:
    ExposureAssessorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.vulnerability_scorer import VulnerabilityScorerEngine
except ImportError:
    VulnerabilityScorerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.compliance_reporter import ComplianceReporterEngine
except ImportError:
    ComplianceReporterEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.climate_hazard.hazard_pipeline import HazardPipelineEngine
except ImportError:
    HazardPipelineEngine = None  # type: ignore[assignment, misc]


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class SourceResponse(BaseModel):
    """Climate hazard data source registration / retrieval response.

    Attributes:
        source_id: Unique source identifier (UUID4).
        name: Human-readable source name.
        source_type: Classification of the data source (noaa, copernicus,
            world_bank, nasa, ipcc, national_agency, satellite,
            ground_station, model_output, custom).
        hazard_types: Hazard types covered by this source.
        status: Source lifecycle status (active, inactive, deprecated).
        region: Geographic region or coverage area.
        description: Human-readable source description.
        metadata: Additional unstructured metadata.
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC last-update timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    source_type: str = Field(default="custom")
    hazard_types: List[str] = Field(default_factory=list)
    status: str = Field(default="active")
    region: str = Field(default="")
    description: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class HazardDataResponse(BaseModel):
    """Hazard data ingestion / query response.

    Attributes:
        record_id: Unique hazard data record identifier (UUID4).
        source_id: Source from which data was ingested.
        hazard_type: Type of climate hazard (flood, drought, wildfire,
            heat_wave, cold_wave, storm, sea_level_rise,
            tropical_cyclone, landslide, water_stress,
            precipitation_change, temperature_change, compound).
        location_id: Location or region identifier for the data.
        scenario: Climate scenario associated with the data.
        value: Primary data value.
        unit: Measurement unit for the value.
        parameters: Additional hazard-specific parameters.
        timestamp_start: ISO-8601 UTC start of observation period.
        timestamp_end: ISO-8601 UTC end of observation period.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = Field(default="")
    hazard_type: str = Field(default="")
    location_id: str = Field(default="")
    scenario: str = Field(default="")
    value: float = Field(default=0.0)
    unit: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp_start: str = Field(default="")
    timestamp_end: str = Field(default="")
    provenance_hash: str = Field(default="")


class HazardEventResponse(BaseModel):
    """Historical climate hazard event response.

    Attributes:
        event_id: Unique event identifier (UUID4).
        hazard_type: Type of climate hazard for this event.
        severity: Event severity level (extreme, high, medium, low,
            negligible).
        location_id: Location or region where the event occurred.
        description: Event description.
        impact_summary: Summary of impact (casualties, damage, area).
        start_date: ISO-8601 UTC event start date.
        end_date: ISO-8601 UTC event end date.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hazard_type: str = Field(default="")
    severity: str = Field(default="medium")
    location_id: str = Field(default="")
    description: str = Field(default="")
    impact_summary: Dict[str, Any] = Field(default_factory=dict)
    start_date: str = Field(default="")
    end_date: str = Field(default="")
    provenance_hash: str = Field(default="")


class RiskIndexResponse(BaseModel):
    """Climate risk index calculation response.

    Attributes:
        index_id: Unique risk index identifier (UUID4).
        location_id: Location for which the risk was calculated.
        hazard_type: Hazard type assessed.
        scenario: Climate scenario used for the calculation.
        composite_score: Computed composite risk score (0-100).
        risk_classification: Risk classification (extreme, high,
            medium, low, negligible).
        component_scores: Individual score components (probability,
            intensity, frequency, duration).
        weights: Weight distribution used for the calculation.
        confidence: Confidence level of the assessment (0.0-1.0).
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    index_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_id: str = Field(default="")
    hazard_type: str = Field(default="")
    scenario: str = Field(default="")
    composite_score: float = Field(default=0.0)
    risk_classification: str = Field(default="negligible")
    component_scores: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class MultiHazardResponse(BaseModel):
    """Multi-hazard composite risk index response.

    Attributes:
        assessment_id: Unique assessment identifier (UUID4).
        location_id: Location for multi-hazard assessment.
        hazard_types: List of hazard types assessed.
        scenario: Climate scenario used.
        composite_score: Overall multi-hazard composite score (0-100).
        risk_classification: Overall risk classification.
        per_hazard_scores: Per-hazard breakdown of scores.
        interaction_effects: Cross-hazard interaction adjustments.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_id: str = Field(default="")
    hazard_types: List[str] = Field(default_factory=list)
    scenario: str = Field(default="")
    composite_score: float = Field(default=0.0)
    risk_classification: str = Field(default="negligible")
    per_hazard_scores: List[Dict[str, Any]] = Field(default_factory=list)
    interaction_effects: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class LocationComparisonResponse(BaseModel):
    """Location risk comparison response.

    Attributes:
        comparison_id: Unique comparison identifier (UUID4).
        location_ids: Locations compared.
        hazard_type: Hazard type compared across locations.
        scenario: Climate scenario used for comparison.
        rankings: Ordered list of locations by risk score.
        per_location_scores: Detailed scores per location.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    comparison_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_ids: List[str] = Field(default_factory=list)
    hazard_type: str = Field(default="")
    scenario: str = Field(default="")
    rankings: List[Dict[str, Any]] = Field(default_factory=list)
    per_location_scores: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ScenarioResponse(BaseModel):
    """Climate scenario projection response.

    Attributes:
        projection_id: Unique projection identifier (UUID4).
        location_id: Location for which the projection was made.
        hazard_type: Hazard type projected.
        scenario: Climate scenario pathway (SSP or RCP).
        time_horizon: Target time horizon (2030, 2050, 2100).
        baseline_value: Baseline (historical) value.
        projected_value: Projected value under the scenario.
        change_percent: Percentage change from baseline.
        confidence_interval: Lower and upper bounds of projection.
        parameters: Additional projection parameters.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    projection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location_id: str = Field(default="")
    hazard_type: str = Field(default="")
    scenario: str = Field(default="SSP2-4.5")
    time_horizon: str = Field(default="MID_TERM")
    baseline_value: float = Field(default=0.0)
    projected_value: float = Field(default=0.0)
    change_percent: float = Field(default=0.0)
    confidence_interval: Dict[str, float] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class AssetResponse(BaseModel):
    """Physical or financial asset registration / retrieval response.

    Attributes:
        asset_id: Unique asset identifier (UUID4).
        name: Human-readable asset name.
        asset_type: Classification of the asset (facility, warehouse,
            office, data_center, factory, supply_chain_node,
            transport_hub, port, mine, farm, renewable_installation,
            portfolio).
        location_id: Location or region identifier for the asset.
        coordinates: Geographic coordinates (latitude, longitude).
        value: Financial value of the asset.
        currency: Currency for the asset value.
        sector: Economic sector of the asset.
        metadata: Additional unstructured metadata.
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    asset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    asset_type: str = Field(default="facility")
    location_id: str = Field(default="")
    coordinates: Dict[str, float] = Field(default_factory=dict)
    value: float = Field(default=0.0)
    currency: str = Field(default="USD")
    sector: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ExposureResponse(BaseModel):
    """Asset-level climate hazard exposure assessment response.

    Attributes:
        exposure_id: Unique exposure assessment identifier (UUID4).
        asset_id: Asset for which exposure was assessed.
        hazard_type: Hazard type assessed.
        scenario: Climate scenario used.
        exposure_level: Exposure classification (extreme, high,
            medium, low, negligible).
        exposure_score: Numeric exposure score (0-100).
        financial_impact: Estimated financial impact.
        impact_details: Detailed breakdown of impact.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    exposure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = Field(default="")
    hazard_type: str = Field(default="")
    scenario: str = Field(default="")
    exposure_level: str = Field(default="negligible")
    exposure_score: float = Field(default=0.0)
    financial_impact: float = Field(default=0.0)
    impact_details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class PortfolioExposureResponse(BaseModel):
    """Portfolio-level climate hazard exposure assessment response.

    Attributes:
        portfolio_id: Unique portfolio assessment identifier (UUID4).
        asset_count: Number of assets in the portfolio.
        total_value: Total financial value of the portfolio.
        currency: Currency for portfolio values.
        overall_exposure_level: Portfolio-wide exposure classification.
        overall_exposure_score: Portfolio-wide exposure score (0-100).
        total_financial_impact: Aggregate financial impact.
        per_asset_exposures: Per-asset exposure details.
        hotspots: Identified high-risk concentration areas.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    portfolio_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_count: int = Field(default=0)
    total_value: float = Field(default=0.0)
    currency: str = Field(default="USD")
    overall_exposure_level: str = Field(default="negligible")
    overall_exposure_score: float = Field(default=0.0)
    total_financial_impact: float = Field(default=0.0)
    per_asset_exposures: List[Dict[str, Any]] = Field(default_factory=list)
    hotspots: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class VulnerabilityResponse(BaseModel):
    """Vulnerability scoring response.

    Attributes:
        vulnerability_id: Unique vulnerability identifier (UUID4).
        entity_id: Entity (asset or location) assessed.
        hazard_type: Hazard type assessed.
        sector: Economic sector of the entity.
        vulnerability_level: Vulnerability classification (extreme,
            high, medium, low, negligible).
        vulnerability_score: Numeric vulnerability score (0-100).
        exposure_score: Exposure component score.
        sensitivity_score: Sensitivity component score.
        adaptive_capacity_score: Adaptive capacity component score.
        residual_risk: Residual risk after adaptation measures.
        recommendations: Adaptation recommendations.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    vulnerability_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = Field(default="")
    hazard_type: str = Field(default="")
    sector: str = Field(default="")
    vulnerability_level: str = Field(default="negligible")
    vulnerability_score: float = Field(default=0.0)
    exposure_score: float = Field(default=0.0)
    sensitivity_score: float = Field(default=0.0)
    adaptive_capacity_score: float = Field(default=0.0)
    residual_risk: float = Field(default=0.0)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ReportResponse(BaseModel):
    """Compliance report generation response.

    Attributes:
        report_id: Unique report identifier (UUID4).
        report_type: Type of report (tcfd, csrd, eu_taxonomy,
            physical_risk, transition_risk, portfolio_summary,
            hotspot_analysis, compliance_summary).
        format: Output format (json, csv, pdf, html, markdown, xml).
        content: Rendered report content or summary.
        report_hash: SHA-256 hash of the full report content.
        generated_at: ISO-8601 UTC generation timestamp.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = Field(default="physical_risk")
    format: str = Field(default="json")
    content: str = Field(default="")
    report_hash: str = Field(default="")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class PipelineResponse(BaseModel):
    """End-to-end climate hazard pipeline execution response.

    Attributes:
        pipeline_id: Unique pipeline run identifier (UUID4).
        stages_completed: Number of pipeline stages completed.
        stages_total: Total number of pipeline stages.
        stage_results: Per-stage execution results.
        overall_status: Pipeline completion status (success, failure,
            partial).
        duration_ms: Total wall-clock pipeline time in milliseconds.
        provenance_hash: SHA-256 provenance hash for audit trail.
    """

    model_config = {"extra": "forbid"}

    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stages_completed: int = Field(default=0)
    stages_total: int = Field(default=7)
    stage_results: List[Dict[str, Any]] = Field(default_factory=list)
    overall_status: str = Field(default="pending")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class HealthResponse(BaseModel):
    """Service health check response.

    Attributes:
        status: Overall service status (healthy, degraded, unhealthy).
        engines: Per-engine availability status.
        engines_available: Count of available engines.
        engines_total: Total number of engines.
        started: Whether the service has been started.
        statistics: Summary statistics.
        provenance_chain_valid: Whether the provenance chain is intact.
        provenance_entries: Total provenance entries recorded.
        prometheus_available: Whether Prometheus client is available.
        timestamp: ISO-8601 UTC timestamp of the health check.
    """

    model_config = {"extra": "forbid"}

    status: str = Field(default="healthy")
    engines: Dict[str, str] = Field(default_factory=dict)
    engines_available: int = Field(default=0)
    engines_total: int = Field(default=7)
    started: bool = Field(default=False)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    provenance_chain_valid: bool = Field(default=True)
    provenance_entries: int = Field(default=0)
    prometheus_available: bool = Field(default=False)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# ClimateHazardService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["ClimateHazardService"] = None


class ClimateHazardService:
    """Unified facade over the Climate Hazard Connector Agent SDK.

    Aggregates all seven climate hazard engines (hazard database, risk
    index, scenario projector, exposure assessor, vulnerability scorer,
    compliance reporter, hazard pipeline) through a single entry point
    with convenience methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: ClimateHazardConfig instance.
        provenance: ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = ClimateHazardService()
        >>> result = service.calculate_risk_index(
        ...     location_id="loc_001",
        ...     hazard_type="flood",
        ...     scenario="SSP2-4.5",
        ... )
        >>> print(result["risk_classification"], result["composite_score"])
    """

    def __init__(
        self,
        config: Optional[ClimateHazardConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize the Climate Hazard Connector Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - HazardDatabaseEngine
        - RiskIndexEngine
        - ScenarioProjectorEngine
        - ExposureAssessorEngine
        - VulnerabilityScorerEngine
        - ComplianceReporterEngine
        - HazardPipelineEngine

        Args:
            config: Optional configuration. Uses global config if None.
            provenance: Optional ProvenanceTracker. Creates a new one
                if None.
        """
        self.config = config if config is not None else get_config()

        # Provenance tracker
        self.provenance = (
            provenance
            if provenance is not None
            else ProvenanceTracker(
                genesis_hash=self.config.genesis_hash,
            )
        )

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._hazard_database_engine: Any = None
        self._risk_index_engine: Any = None
        self._scenario_projector_engine: Any = None
        self._exposure_assessor_engine: Any = None
        self._vulnerability_scorer_engine: Any = None
        self._compliance_reporter_engine: Any = None
        self._hazard_pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._sources: Dict[str, SourceResponse] = {}
        self._hazard_data: Dict[str, HazardDataResponse] = {}
        self._hazard_events: Dict[str, HazardEventResponse] = {}
        self._risk_indices: Dict[str, RiskIndexResponse] = {}
        self._multi_hazards: Dict[str, MultiHazardResponse] = {}
        self._comparisons: Dict[str, LocationComparisonResponse] = {}
        self._scenarios: Dict[str, ScenarioResponse] = {}
        self._assets: Dict[str, AssetResponse] = {}
        self._exposures: Dict[str, ExposureResponse] = {}
        self._portfolio_exposures: Dict[str, PortfolioExposureResponse] = {}
        self._vulnerabilities: Dict[str, VulnerabilityResponse] = {}
        self._reports: Dict[str, ReportResponse] = {}
        self._pipeline_results: Dict[str, PipelineResponse] = {}

        # Statistics counters
        self._total_ingestions: int = 0
        self._total_risk_calculations: int = 0
        self._total_projections: int = 0
        self._total_exposure_assessments: int = 0
        self._total_vulnerability_scores: int = 0
        self._total_reports: int = 0
        self._total_pipeline_runs: int = 0
        self._started: bool = False

        logger.info("ClimateHazardService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def hazard_database_engine(self) -> Any:
        """Get the HazardDatabaseEngine instance."""
        return self._hazard_database_engine

    @property
    def risk_index_engine(self) -> Any:
        """Get the RiskIndexEngine instance."""
        return self._risk_index_engine

    @property
    def scenario_projector_engine(self) -> Any:
        """Get the ScenarioProjectorEngine instance."""
        return self._scenario_projector_engine

    @property
    def exposure_assessor_engine(self) -> Any:
        """Get the ExposureAssessorEngine instance."""
        return self._exposure_assessor_engine

    @property
    def vulnerability_scorer_engine(self) -> Any:
        """Get the VulnerabilityScorerEngine instance."""
        return self._vulnerability_scorer_engine

    @property
    def compliance_reporter_engine(self) -> Any:
        """Get the ComplianceReporterEngine instance."""
        return self._compliance_reporter_engine

    @property
    def hazard_pipeline_engine(self) -> Any:
        """Get the HazardPipelineEngine instance."""
        return self._hazard_pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are wired together using dependency injection: the
        database engine is shared with the risk index engine, the risk
        index engine is shared with the scenario projector and exposure
        assessor, and so on. The shared ProvenanceTracker is injected
        into all engines for unified audit trails.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        # E1: HazardDatabaseEngine(provenance)
        if HazardDatabaseEngine is not None:
            try:
                self._hazard_database_engine = HazardDatabaseEngine(
                    provenance=self.provenance,
                )
                logger.info("HazardDatabaseEngine initialized")
            except Exception as exc:
                logger.warning("HazardDatabaseEngine init failed: %s", exc)
        else:
            logger.warning("HazardDatabaseEngine not available; using stub")

        # E2: RiskIndexEngine(database, provenance)
        if RiskIndexEngine is not None:
            try:
                self._risk_index_engine = RiskIndexEngine(
                    database=self._hazard_database_engine,
                    provenance=self.provenance,
                )
                logger.info("RiskIndexEngine initialized")
            except Exception as exc:
                logger.warning("RiskIndexEngine init failed: %s", exc)
        else:
            logger.warning("RiskIndexEngine not available; using stub")

        # E3: ScenarioProjectorEngine(risk_engine, provenance)
        if ScenarioProjectorEngine is not None:
            try:
                self._scenario_projector_engine = ScenarioProjectorEngine(
                    risk_engine=self._risk_index_engine,
                    provenance=self.provenance,
                )
                logger.info("ScenarioProjectorEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ScenarioProjectorEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ScenarioProjectorEngine not available; using stub"
            )

        # E4: ExposureAssessorEngine(risk_engine, provenance)
        if ExposureAssessorEngine is not None:
            try:
                self._exposure_assessor_engine = ExposureAssessorEngine(
                    risk_engine=self._risk_index_engine,
                    provenance=self.provenance,
                )
                logger.info("ExposureAssessorEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ExposureAssessorEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ExposureAssessorEngine not available; using stub"
            )

        # E5: VulnerabilityScorerEngine(exposure_engine, provenance)
        if VulnerabilityScorerEngine is not None:
            try:
                self._vulnerability_scorer_engine = VulnerabilityScorerEngine(
                    exposure_engine=self._exposure_assessor_engine,
                    provenance=self.provenance,
                )
                logger.info("VulnerabilityScorerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "VulnerabilityScorerEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "VulnerabilityScorerEngine not available; using stub"
            )

        # E6: ComplianceReporterEngine(provenance)
        if ComplianceReporterEngine is not None:
            try:
                self._compliance_reporter_engine = ComplianceReporterEngine(
                    provenance=self.provenance,
                )
                logger.info("ComplianceReporterEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ComplianceReporterEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ComplianceReporterEngine not available; using stub"
            )

        # E7: HazardPipelineEngine(database, risk_engine, projector,
        #     exposure_engine, vulnerability_engine, reporter, provenance)
        if HazardPipelineEngine is not None:
            try:
                self._hazard_pipeline_engine = HazardPipelineEngine(
                    database=self._hazard_database_engine,
                    risk_engine=self._risk_index_engine,
                    projector=self._scenario_projector_engine,
                    exposure_engine=self._exposure_assessor_engine,
                    vulnerability_engine=self._vulnerability_scorer_engine,
                    reporter=self._compliance_reporter_engine,
                    provenance=self.provenance,
                )
                logger.info("HazardPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "HazardPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "HazardPipelineEngine not available; using stub"
            )

    # ==================================================================
    # Source operations (delegate to HazardDatabaseEngine)
    # ==================================================================

    def register_source(
        self,
        name: str = "",
        source_type: str = "custom",
        hazard_types: Optional[List[str]] = None,
        region: str = "",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Register a new climate hazard data source.

        Delegates to the HazardDatabaseEngine for source registration.
        All operations are deterministic (zero-hallucination).

        Args:
            name: Human-readable source name.
            source_type: Data source classification.
            hazard_types: Hazard types covered by this source.
            region: Geographic region or coverage area.
            description: Source description.
            metadata: Additional metadata.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with registered source details.

        Raises:
            ValueError: If name is empty.
        """
        t0 = time.perf_counter()

        if not name:
            raise ValueError("name must not be empty")

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._hazard_database_engine is not None:
                try:
                    engine_result = self._hazard_database_engine.register_source(
                        name=name,
                        source_type=source_type,
                        hazard_types=hazard_types or [],
                        region=region,
                        description=description,
                        metadata=metadata or {},
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    # Engine may not accept all facade-level parameters;
                    # fall back to in-memory handling.
                    try:
                        engine_result = self._hazard_database_engine.register_source(
                            source_id=_new_uuid(),
                            name=name,
                            source_type=source_type.upper() if source_type else "CUSTOM",
                            hazard_types=hazard_types or [],
                        )
                    except (AttributeError, TypeError, ValueError):
                        pass

            # Build response
            source_id = (
                engine_result.get("source_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = SourceResponse(
                source_id=source_id,
                name=name,
                source_type=source_type,
                hazard_types=hazard_types or [],
                status=engine_result.get("status", "active") if engine_result else "active",
                region=region,
                description=description,
                metadata=metadata or {},
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
                updated_at=engine_result.get("updated_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._sources[response.source_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="hazard_source",
                action="register_source",
                entity_id=response.source_id,
                metadata={
                    "name": name,
                    "source_type": source_type,
                    "hazard_types": hazard_types or [],
                },
            )

            # Record metrics
            set_active_sources(len(self._sources))
            elapsed = time.perf_counter() - t0
            observe_ingestion_duration(source_type, elapsed)

            logger.info(
                "Registered source %s: name=%s type=%s hazard_types=%s",
                response.source_id,
                name,
                source_type,
                hazard_types or [],
            )
            return response.model_dump()

        except Exception as exc:
            logger.error("register_source failed: %s", exc, exc_info=True)
            raise

    def list_sources(
        self,
        hazard_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List registered climate hazard data sources with filtering.

        Args:
            hazard_type: Filter by hazard type coverage.
            status: Filter by source lifecycle status.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of source dictionaries.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._hazard_database_engine is not None:
                try:
                    engine_results = self._hazard_database_engine.list_sources(
                        hazard_type=hazard_type,
                        status=status,
                        limit=limit,
                        offset=offset,
                    )
                    elapsed = time.perf_counter() - t0
                    return engine_results if isinstance(engine_results, list) else []
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store
            sources = list(self._sources.values())
            if hazard_type is not None:
                sources = [
                    s for s in sources
                    if hazard_type in s.hazard_types
                ]
            if status is not None:
                sources = [s for s in sources if s.status == status]
            paginated = sources[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            return [s.model_dump() for s in paginated]

        except Exception as exc:
            logger.error("list_sources failed: %s", exc, exc_info=True)
            raise

    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get a climate hazard source by its unique identifier.

        Args:
            source_id: Source identifier (UUID4 string).

        Returns:
            Source dictionary or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._hazard_database_engine is not None:
                try:
                    engine_result = self._hazard_database_engine.get_source(
                        source_id,
                    )
                    if engine_result is not None:
                        elapsed = time.perf_counter() - t0
                        return engine_result if isinstance(engine_result, dict) else None
                except (AttributeError, TypeError, KeyError):
                    pass

            # Fallback to in-memory store
            cached = self._sources.get(source_id)
            elapsed = time.perf_counter() - t0
            return cached.model_dump() if cached is not None else None

        except Exception as exc:
            logger.error("get_source failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Hazard data operations (delegate to HazardDatabaseEngine)
    # ==================================================================

    def ingest_hazard_data(
        self,
        source_id: str = "",
        hazard_type: str = "",
        location_id: str = "",
        scenario: str = "",
        value: float = 0.0,
        unit: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        timestamp_start: str = "",
        timestamp_end: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Ingest climate hazard data from a registered source.

        Delegates to the HazardDatabaseEngine for data ingestion. All
        values are stored deterministically (zero-hallucination).

        Args:
            source_id: Source from which data is ingested.
            hazard_type: Type of climate hazard.
            location_id: Location identifier for the data.
            scenario: Climate scenario associated with the data.
            value: Primary data value.
            unit: Measurement unit.
            parameters: Additional hazard-specific parameters.
            timestamp_start: ISO-8601 UTC start of observation period.
            timestamp_end: ISO-8601 UTC end of observation period.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with ingested data record details.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._hazard_database_engine is not None:
                try:
                    engine_result = self._hazard_database_engine.ingest_data(
                        source_id=source_id,
                        hazard_type=hazard_type,
                        location_id=location_id,
                        scenario=scenario,
                        value=value,
                        unit=unit,
                        parameters=parameters or {},
                        timestamp_start=timestamp_start,
                        timestamp_end=timestamp_end,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    # Engine may use a different method name or signature
                    try:
                        engine_result = self._hazard_database_engine.ingest_hazard_data(
                            source_id=source_id,
                            hazard_type=hazard_type.upper() if hazard_type else "",
                            records=[{
                                "location": {"lat": 0.0, "lon": 0.0},
                                "intensity": min(value / 10.0, 10.0),
                                "probability": 0.5,
                            }],
                        )
                    except (AttributeError, TypeError, ValueError, KeyError):
                        pass

            # Build response
            record_id = (
                engine_result.get("record_id", _new_uuid())
                if engine_result else _new_uuid()
            )

            response = HazardDataResponse(
                record_id=record_id,
                source_id=source_id,
                hazard_type=hazard_type,
                location_id=location_id,
                scenario=scenario,
                value=value,
                unit=unit,
                parameters=parameters or {},
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._hazard_data[response.record_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="hazard_data",
                action="ingest_data",
                entity_id=response.record_id,
                metadata={
                    "source_id": source_id,
                    "hazard_type": hazard_type,
                    "location_id": location_id,
                },
            )

            # Record metrics
            elapsed = time.perf_counter() - t0
            source_type = "custom"
            src = self._sources.get(source_id)
            if src is not None:
                source_type = src.source_type
            record_ingestion(hazard_type, source_type)
            observe_ingestion_duration(source_type, elapsed)
            self._total_ingestions += 1

            logger.info(
                "Ingested hazard data %s: type=%s source=%s location=%s",
                response.record_id,
                hazard_type,
                source_id,
                location_id,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "ingest_hazard_data failed: %s", exc, exc_info=True,
            )
            raise

    def query_hazard_data(
        self,
        hazard_type: Optional[str] = None,
        source_id: Optional[str] = None,
        location_id: Optional[str] = None,
        scenario: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query ingested climate hazard data with optional filters.

        Args:
            hazard_type: Filter by hazard type.
            source_id: Filter by source identifier.
            location_id: Filter by location identifier.
            scenario: Filter by climate scenario.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of hazard data record dictionaries.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._hazard_database_engine is not None:
                try:
                    engine_results = self._hazard_database_engine.query_data(
                        hazard_type=hazard_type,
                        source_id=source_id,
                        location_id=location_id,
                        scenario=scenario,
                        limit=limit,
                        offset=offset,
                    )
                    elapsed = time.perf_counter() - t0
                    return engine_results if isinstance(engine_results, list) else []
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store
            records = list(self._hazard_data.values())
            if hazard_type is not None:
                records = [r for r in records if r.hazard_type == hazard_type]
            if source_id is not None:
                records = [r for r in records if r.source_id == source_id]
            if location_id is not None:
                records = [r for r in records if r.location_id == location_id]
            if scenario is not None:
                records = [r for r in records if r.scenario == scenario]
            paginated = records[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            return [r.model_dump() for r in paginated]

        except Exception as exc:
            logger.error(
                "query_hazard_data failed: %s", exc, exc_info=True,
            )
            raise

    def list_hazard_events(
        self,
        hazard_type: Optional[str] = None,
        severity: Optional[str] = None,
        location_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List historical climate hazard events with optional filters.

        Args:
            hazard_type: Filter by hazard type.
            severity: Filter by event severity.
            location_id: Filter by location identifier.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of hazard event dictionaries.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine if available
            if self._hazard_database_engine is not None:
                try:
                    engine_results = self._hazard_database_engine.list_events(
                        hazard_type=hazard_type,
                        severity=severity,
                        location_id=location_id,
                        limit=limit,
                        offset=offset,
                    )
                    elapsed = time.perf_counter() - t0
                    return engine_results if isinstance(engine_results, list) else []
                except (AttributeError, TypeError):
                    pass

            # Fallback to in-memory store
            events = list(self._hazard_events.values())
            if hazard_type is not None:
                events = [e for e in events if e.hazard_type == hazard_type]
            if severity is not None:
                events = [e for e in events if e.severity == severity]
            if location_id is not None:
                events = [e for e in events if e.location_id == location_id]
            paginated = events[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            return [e.model_dump() for e in paginated]

        except Exception as exc:
            logger.error(
                "list_hazard_events failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Risk index operations (delegate to RiskIndexEngine)
    # ==================================================================

    def calculate_risk_index(
        self,
        location_id: str = "",
        hazard_type: str = "",
        scenario: str = "",
        probability: float = 0.0,
        intensity: float = 0.0,
        frequency: float = 0.0,
        duration: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate a composite climate risk index for a location.

        Delegates to the RiskIndexEngine. All calculations are
        deterministic (zero-hallucination) using weighted arithmetic.

        Args:
            location_id: Location for which to calculate risk.
            hazard_type: Type of climate hazard.
            scenario: Climate scenario used for calculation.
            probability: Hazard probability score (0-100).
            intensity: Hazard intensity score (0-100).
            frequency: Hazard frequency score (0-100).
            duration: Hazard duration score (0-100).
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with risk index calculation results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._risk_index_engine is not None:
                try:
                    engine_result = self._risk_index_engine.calculate_risk(
                        location_id=location_id,
                        hazard_type=hazard_type,
                        scenario=scenario,
                        probability=probability,
                        intensity=intensity,
                        frequency=frequency,
                        duration=duration,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    # Engine may use a different method name or signature;
                    # engine expects probability in [0,1] and intensity in
                    # [0,10] while facade uses [0,100] scale for both.
                    try:
                        engine_result = self._risk_index_engine.calculate_risk_index(
                            hazard_type=hazard_type,
                            location={"location_id": location_id},
                            probability=probability / 100.0,
                            intensity=intensity / 10.0,
                            frequency=frequency / 100.0,
                            duration_days=max(duration, 1.0),
                            scenario=scenario or None,
                        )
                    except (AttributeError, TypeError, ValueError):
                        pass

            # Build response
            index_id = (
                engine_result.get("index_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            composite_score = (
                engine_result.get("composite_score", 0.0)
                if engine_result else 0.0
            )
            risk_classification = (
                engine_result.get("risk_classification", "negligible")
                if engine_result else "negligible"
            )
            raw_components = (
                engine_result.get("component_scores", {})
                if engine_result else {}
            )
            # Flatten component_scores: engine may return nested dicts
            # like {"probability": {"raw": 0.65, "normalised": 0.65, "weight": 0.3}}
            # but Pydantic expects Dict[str, float]
            component_scores: Dict[str, float] = {}
            for k, v in raw_components.items():
                if isinstance(v, dict):
                    component_scores[k] = float(v.get("normalised", v.get("raw", 0.0)))
                else:
                    component_scores[k] = float(v)
            weights = (
                engine_result.get("weights", {})
                if engine_result else {}
            )
            confidence = (
                engine_result.get("confidence", 0.0)
                if engine_result else 0.0
            )

            response = RiskIndexResponse(
                index_id=index_id,
                location_id=location_id,
                hazard_type=hazard_type,
                scenario=scenario,
                composite_score=composite_score,
                risk_classification=risk_classification,
                component_scores=component_scores,
                weights=weights,
                confidence=confidence,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._risk_indices[response.index_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="risk_index",
                action="calculate_risk",
                entity_id=response.index_id,
                metadata={
                    "location_id": location_id,
                    "hazard_type": hazard_type,
                    "scenario": scenario,
                    "composite_score": composite_score,
                    "risk_classification": risk_classification,
                },
            )

            # Record metrics
            record_risk_calculation(hazard_type, scenario)
            self._total_risk_calculations += 1

            # Update high-risk gauge
            high_risk_count = sum(
                1 for r in self._risk_indices.values()
                if r.risk_classification in ("extreme", "high")
            )
            set_high_risk(high_risk_count)

            elapsed = time.perf_counter() - t0

            logger.info(
                "Calculated risk index %s: location=%s hazard=%s "
                "score=%.2f classification=%s",
                response.index_id,
                location_id,
                hazard_type,
                composite_score,
                risk_classification,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "calculate_risk_index failed: %s", exc, exc_info=True,
            )
            raise

    def calculate_multi_hazard(
        self,
        location_id: str = "",
        hazard_types: Optional[List[str]] = None,
        scenario: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate a multi-hazard composite risk index.

        Delegates to the RiskIndexEngine for multi-hazard assessment.

        Args:
            location_id: Location for multi-hazard assessment.
            hazard_types: List of hazard types to assess.
            scenario: Climate scenario used.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with multi-hazard assessment results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._risk_index_engine is not None:
                try:
                    engine_result = self._risk_index_engine.calculate_multi_hazard(
                        location_id=location_id,
                        hazard_types=hazard_types or [],
                        scenario=scenario,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            assessment_id = (
                engine_result.get("assessment_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            composite_score = (
                engine_result.get("composite_score", 0.0)
                if engine_result else 0.0
            )
            risk_classification = (
                engine_result.get("risk_classification", "negligible")
                if engine_result else "negligible"
            )

            response = MultiHazardResponse(
                assessment_id=assessment_id,
                location_id=location_id,
                hazard_types=hazard_types or [],
                scenario=scenario,
                composite_score=composite_score,
                risk_classification=risk_classification,
                per_hazard_scores=engine_result.get("per_hazard_scores", []) if engine_result else [],
                interaction_effects=engine_result.get("interaction_effects", {}) if engine_result else {},
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._multi_hazards[response.assessment_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="risk_index",
                action="calculate_multi_hazard",
                entity_id=response.assessment_id,
                metadata={
                    "location_id": location_id,
                    "hazard_types": hazard_types or [],
                    "composite_score": composite_score,
                },
            )

            # Record metrics
            for ht in (hazard_types or []):
                record_risk_calculation(ht, scenario)

            elapsed = time.perf_counter() - t0

            logger.info(
                "Multi-hazard assessment %s: location=%s hazards=%s "
                "score=%.2f",
                response.assessment_id,
                location_id,
                hazard_types or [],
                composite_score,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "calculate_multi_hazard failed: %s", exc, exc_info=True,
            )
            raise

    def compare_locations(
        self,
        location_ids: Optional[List[str]] = None,
        hazard_type: str = "",
        scenario: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compare climate risk indices across multiple locations.

        Delegates to the RiskIndexEngine for location comparison.

        Args:
            location_ids: Locations to compare.
            hazard_type: Hazard type to compare across locations.
            scenario: Climate scenario used.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with location comparison results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._risk_index_engine is not None:
                try:
                    engine_result = self._risk_index_engine.compare_locations(
                        location_ids=location_ids or [],
                        hazard_type=hazard_type,
                        scenario=scenario,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            comparison_id = (
                engine_result.get("comparison_id", _new_uuid())
                if engine_result else _new_uuid()
            )

            response = LocationComparisonResponse(
                comparison_id=comparison_id,
                location_ids=location_ids or [],
                hazard_type=hazard_type,
                scenario=scenario,
                rankings=engine_result.get("rankings", []) if engine_result else [],
                per_location_scores=engine_result.get("per_location_scores", []) if engine_result else [],
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._comparisons[response.comparison_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="risk_index",
                action="rank_hazards",
                entity_id=response.comparison_id,
                metadata={
                    "location_ids": location_ids or [],
                    "hazard_type": hazard_type,
                },
            )

            elapsed = time.perf_counter() - t0

            logger.info(
                "Location comparison %s: locations=%s hazard=%s",
                response.comparison_id,
                location_ids or [],
                hazard_type,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "compare_locations failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Scenario projection operations (delegate to ScenarioProjectorEngine)
    # ==================================================================

    def project_scenario(
        self,
        location_id: str = "",
        hazard_type: str = "",
        scenario: str = "SSP2-4.5",
        time_horizon: str = "MID_TERM",
        baseline_value: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Project climate hazard under a given SSP or RCP scenario.

        Delegates to the ScenarioProjectorEngine. All projections use
        deterministic scaling formulas (zero-hallucination).

        Args:
            location_id: Location for the projection.
            hazard_type: Type of climate hazard to project.
            scenario: Climate scenario pathway (SSP or RCP).
            time_horizon: Target time horizon (SHORT_TERM, MID_TERM,
                LONG_TERM).
            baseline_value: Historical baseline value.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with projection results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._scenario_projector_engine is not None:
                try:
                    engine_result = self._scenario_projector_engine.project_scenario(
                        location_id=location_id,
                        hazard_type=hazard_type,
                        scenario=scenario,
                        time_horizon=time_horizon,
                        baseline_value=baseline_value,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    # Engine may use a different method name or signature;
                    # engine expects UPPER_CASE hazard types like
                    # RIVERINE_FLOOD, EXTREME_HEAT, etc.
                    try:
                        engine_result = self._scenario_projector_engine.project_hazard(
                            hazard_type=hazard_type.upper(),
                            location={"location_id": location_id},
                            baseline_risk={"value": baseline_value},
                            scenario=scenario,
                            time_horizon=time_horizon,
                        )
                    except (AttributeError, TypeError, ValueError):
                        pass

            # Build response
            projection_id = (
                engine_result.get("projection_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            projected_value = (
                engine_result.get("projected_value", 0.0)
                if engine_result else 0.0
            )
            change_percent = (
                engine_result.get("change_percent", 0.0)
                if engine_result else 0.0
            )

            response = ScenarioResponse(
                projection_id=projection_id,
                location_id=location_id,
                hazard_type=hazard_type,
                scenario=scenario,
                time_horizon=time_horizon,
                baseline_value=baseline_value,
                projected_value=projected_value,
                change_percent=change_percent,
                confidence_interval=engine_result.get("confidence_interval", {}) if engine_result else {},
                parameters=engine_result.get("parameters", {}) if engine_result else {},
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._scenarios[response.projection_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="scenario_projection",
                action="project_scenario",
                entity_id=response.projection_id,
                metadata={
                    "location_id": location_id,
                    "hazard_type": hazard_type,
                    "scenario": scenario,
                    "time_horizon": time_horizon,
                },
            )

            # Record metrics
            record_projection(scenario, time_horizon)
            self._total_projections += 1
            elapsed = time.perf_counter() - t0

            logger.info(
                "Scenario projection %s: location=%s hazard=%s "
                "scenario=%s horizon=%s projected=%.2f change=%.2f%%",
                response.projection_id,
                location_id,
                hazard_type,
                scenario,
                time_horizon,
                projected_value,
                change_percent,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "project_scenario failed: %s", exc, exc_info=True,
            )
            raise

    def list_scenarios(
        self,
        scenario: Optional[str] = None,
        time_horizon: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List available or previously computed scenario projections.

        Args:
            scenario: Filter by climate scenario pathway.
            time_horizon: Filter by time horizon.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of scenario projection dictionaries.
        """
        t0 = time.perf_counter()

        try:
            # Fallback to in-memory store
            projections = list(self._scenarios.values())
            if scenario is not None:
                projections = [
                    p for p in projections if p.scenario == scenario
                ]
            if time_horizon is not None:
                projections = [
                    p for p in projections if p.time_horizon == time_horizon
                ]
            paginated = projections[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            return [p.model_dump() for p in paginated]

        except Exception as exc:
            logger.error(
                "list_scenarios failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Asset operations (delegate to ExposureAssessorEngine)
    # ==================================================================

    def register_asset(
        self,
        name: str = "",
        asset_type: str = "facility",
        location_id: str = "",
        coordinates: Optional[Dict[str, float]] = None,
        value: float = 0.0,
        currency: str = "USD",
        sector: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Register a physical or financial asset for climate hazard monitoring.

        Delegates to the ExposureAssessorEngine for asset registration.

        Args:
            name: Human-readable asset name.
            asset_type: Asset classification.
            location_id: Location identifier.
            coordinates: Geographic coordinates (latitude, longitude).
            value: Financial value of the asset.
            currency: Currency for the asset value.
            sector: Economic sector.
            metadata: Additional metadata.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with registered asset details.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._exposure_assessor_engine is not None:
                try:
                    engine_result = self._exposure_assessor_engine.register_asset(
                        name=name,
                        asset_type=asset_type,
                        location_id=location_id,
                        coordinates=coordinates or {},
                        value=value,
                        currency=currency,
                        sector=sector,
                        metadata=metadata or {},
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            asset_id = (
                engine_result.get("asset_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            now_iso = _utcnow_iso()

            response = AssetResponse(
                asset_id=asset_id,
                name=name,
                asset_type=asset_type,
                location_id=location_id,
                coordinates=coordinates or {},
                value=value,
                currency=currency,
                sector=sector,
                metadata=metadata or {},
                created_at=engine_result.get("created_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._assets[response.asset_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="asset",
                action="register_asset",
                entity_id=response.asset_id,
                metadata={
                    "name": name,
                    "asset_type": asset_type,
                    "location_id": location_id,
                },
            )

            # Record metrics
            set_active_assets(len(self._assets))
            elapsed = time.perf_counter() - t0

            logger.info(
                "Registered asset %s: name=%s type=%s location=%s",
                response.asset_id,
                name,
                asset_type,
                location_id,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error("register_asset failed: %s", exc, exc_info=True)
            raise

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        location_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List registered assets with optional filters.

        Args:
            asset_type: Filter by asset type.
            location_id: Filter by location identifier.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of asset dictionaries.
        """
        t0 = time.perf_counter()

        try:
            # Fallback to in-memory store
            assets = list(self._assets.values())
            if asset_type is not None:
                assets = [a for a in assets if a.asset_type == asset_type]
            if location_id is not None:
                assets = [a for a in assets if a.location_id == location_id]
            paginated = assets[offset:offset + limit]

            elapsed = time.perf_counter() - t0
            return [a.model_dump() for a in paginated]

        except Exception as exc:
            logger.error("list_assets failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Exposure assessment operations (delegate to ExposureAssessorEngine)
    # ==================================================================

    def assess_exposure(
        self,
        asset_id: str = "",
        hazard_type: str = "",
        scenario: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Assess climate hazard exposure for an asset.

        Delegates to the ExposureAssessorEngine. All calculations are
        deterministic (zero-hallucination).

        Args:
            asset_id: Asset for which to assess exposure.
            hazard_type: Type of climate hazard.
            scenario: Climate scenario used.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with exposure assessment results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._exposure_assessor_engine is not None:
                try:
                    engine_result = self._exposure_assessor_engine.assess_exposure(
                        asset_id=asset_id,
                        hazard_type=hazard_type,
                        scenario=scenario,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            exposure_id = (
                engine_result.get("exposure_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            exposure_level = (
                engine_result.get("exposure_level", "negligible")
                if engine_result else "negligible"
            )
            exposure_score = (
                engine_result.get("exposure_score", 0.0)
                if engine_result else 0.0
            )

            response = ExposureResponse(
                exposure_id=exposure_id,
                asset_id=asset_id,
                hazard_type=hazard_type,
                scenario=scenario,
                exposure_level=exposure_level,
                exposure_score=exposure_score,
                financial_impact=engine_result.get("financial_impact", 0.0) if engine_result else 0.0,
                impact_details=engine_result.get("impact_details", {}) if engine_result else {},
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._exposures[response.exposure_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="exposure",
                action="assess_exposure",
                entity_id=response.exposure_id,
                metadata={
                    "asset_id": asset_id,
                    "hazard_type": hazard_type,
                    "exposure_level": exposure_level,
                },
            )

            # Record metrics
            asset_type = "facility"
            cached_asset = self._assets.get(asset_id)
            if cached_asset is not None:
                asset_type = cached_asset.asset_type
            record_exposure(asset_type, hazard_type)
            self._total_exposure_assessments += 1
            elapsed = time.perf_counter() - t0

            logger.info(
                "Exposure assessment %s: asset=%s hazard=%s "
                "level=%s score=%.2f",
                response.exposure_id,
                asset_id,
                hazard_type,
                exposure_level,
                exposure_score,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "assess_exposure failed: %s", exc, exc_info=True,
            )
            raise

    def assess_portfolio_exposure(
        self,
        asset_ids: Optional[List[str]] = None,
        hazard_type: str = "",
        scenario: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Assess climate hazard exposure for an entire asset portfolio.

        Delegates to the ExposureAssessorEngine for portfolio analysis.

        Args:
            asset_ids: List of asset identifiers in the portfolio.
            hazard_type: Type of climate hazard.
            scenario: Climate scenario used.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with portfolio exposure assessment results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._exposure_assessor_engine is not None:
                try:
                    engine_result = self._exposure_assessor_engine.assess_portfolio(
                        asset_ids=asset_ids or [],
                        hazard_type=hazard_type,
                        scenario=scenario,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            portfolio_id = (
                engine_result.get("portfolio_id", _new_uuid())
                if engine_result else _new_uuid()
            )

            response = PortfolioExposureResponse(
                portfolio_id=portfolio_id,
                asset_count=engine_result.get("asset_count", len(asset_ids or [])) if engine_result else len(asset_ids or []),
                total_value=engine_result.get("total_value", 0.0) if engine_result else 0.0,
                currency=engine_result.get("currency", "USD") if engine_result else "USD",
                overall_exposure_level=engine_result.get("overall_exposure_level", "negligible") if engine_result else "negligible",
                overall_exposure_score=engine_result.get("overall_exposure_score", 0.0) if engine_result else 0.0,
                total_financial_impact=engine_result.get("total_financial_impact", 0.0) if engine_result else 0.0,
                per_asset_exposures=engine_result.get("per_asset_exposures", []) if engine_result else [],
                hotspots=engine_result.get("hotspots", []) if engine_result else [],
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._portfolio_exposures[response.portfolio_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="exposure",
                action="assess_portfolio",
                entity_id=response.portfolio_id,
                metadata={
                    "asset_count": len(asset_ids or []),
                    "hazard_type": hazard_type,
                },
            )

            # Record metrics
            record_exposure("portfolio", hazard_type)
            elapsed = time.perf_counter() - t0

            logger.info(
                "Portfolio exposure %s: assets=%d hazard=%s",
                response.portfolio_id,
                len(asset_ids or []),
                hazard_type,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "assess_portfolio_exposure failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Vulnerability scoring (delegate to VulnerabilityScorerEngine)
    # ==================================================================

    def score_vulnerability(
        self,
        entity_id: str = "",
        hazard_type: str = "",
        sector: str = "",
        exposure_score: float = 0.0,
        sensitivity_score: float = 0.0,
        adaptive_capacity_score: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Score climate vulnerability for an asset or entity.

        Delegates to the VulnerabilityScorerEngine. All calculations
        use deterministic weighted formulas (zero-hallucination).

        Args:
            entity_id: Entity (asset or location) to assess.
            hazard_type: Type of climate hazard.
            sector: Economic sector.
            exposure_score: Exposure component score (0-100).
            sensitivity_score: Sensitivity component score (0-100).
            adaptive_capacity_score: Adaptive capacity score (0-100).
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with vulnerability scoring results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._vulnerability_scorer_engine is not None:
                try:
                    engine_result = self._vulnerability_scorer_engine.score_vulnerability(
                        entity_id=entity_id,
                        hazard_type=hazard_type,
                        sector=sector,
                        exposure_score=exposure_score,
                        sensitivity_score=sensitivity_score,
                        adaptive_capacity_score=adaptive_capacity_score,
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            vulnerability_id = (
                engine_result.get("vulnerability_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            vuln_score = (
                engine_result.get("vulnerability_score", 0.0)
                if engine_result else 0.0
            )
            vulnerability_level = (
                engine_result.get("vulnerability_level", "negligible")
                if engine_result else "negligible"
            )

            response = VulnerabilityResponse(
                vulnerability_id=vulnerability_id,
                entity_id=entity_id,
                hazard_type=hazard_type,
                sector=sector,
                vulnerability_level=vulnerability_level,
                vulnerability_score=vuln_score,
                exposure_score=engine_result.get("exposure_score", exposure_score) if engine_result else exposure_score,
                sensitivity_score=engine_result.get("sensitivity_score", sensitivity_score) if engine_result else sensitivity_score,
                adaptive_capacity_score=engine_result.get("adaptive_capacity_score", adaptive_capacity_score) if engine_result else adaptive_capacity_score,
                residual_risk=engine_result.get("residual_risk", 0.0) if engine_result else 0.0,
                recommendations=engine_result.get("recommendations", []) if engine_result else [],
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._vulnerabilities[response.vulnerability_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="vulnerability",
                action="score_vulnerability",
                entity_id=response.vulnerability_id,
                metadata={
                    "entity_id": entity_id,
                    "hazard_type": hazard_type,
                    "sector": sector,
                    "vulnerability_score": vuln_score,
                    "vulnerability_level": vulnerability_level,
                },
            )

            # Record metrics
            record_vulnerability(sector or "unknown", hazard_type)
            self._total_vulnerability_scores += 1
            elapsed = time.perf_counter() - t0

            logger.info(
                "Vulnerability score %s: entity=%s hazard=%s "
                "level=%s score=%.2f",
                response.vulnerability_id,
                entity_id,
                hazard_type,
                vulnerability_level,
                vuln_score,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "score_vulnerability failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # Compliance reporting (delegate to ComplianceReporterEngine)
    # ==================================================================

    def generate_report(
        self,
        report_type: str = "physical_risk",
        format: str = "json",
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a climate hazard compliance report.

        Delegates to the ComplianceReporterEngine for report generation.
        Supports TCFD, CSRD/ESRS, EU Taxonomy, and custom report types.

        Args:
            report_type: Type of report to generate (tcfd, csrd,
                eu_taxonomy, physical_risk, transition_risk,
                portfolio_summary, hotspot_analysis,
                compliance_summary).
            format: Output format (json, csv, pdf, html, markdown, xml).
            parameters: Additional report-specific parameters.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with generated report details.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._compliance_reporter_engine is not None:
                try:
                    engine_result = self._compliance_reporter_engine.generate_report(
                        report_type=report_type,
                        format=format,
                        parameters=parameters or {},
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            report_id = (
                engine_result.get("report_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            content = (
                engine_result.get("content", "")
                if engine_result else ""
            )
            report_hash = (
                engine_result.get("report_hash", "")
                if engine_result else ""
            )
            if not report_hash and content:
                report_hash = hashlib.sha256(content.encode()).hexdigest()

            now_iso = _utcnow_iso()

            response = ReportResponse(
                report_id=report_id,
                report_type=report_type,
                format=format,
                content=content,
                report_hash=report_hash,
                generated_at=engine_result.get("generated_at", now_iso) if engine_result else now_iso,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._reports[response.report_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="compliance_report",
                action="generate_report",
                entity_id=response.report_id,
                metadata={
                    "report_type": report_type,
                    "format": format,
                },
            )

            # Record metrics
            record_report(report_type, format)
            self._total_reports += 1
            elapsed = time.perf_counter() - t0

            logger.info(
                "Generated report %s: type=%s format=%s",
                response.report_id,
                report_type,
                format,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error(
                "generate_report failed: %s", exc, exc_info=True,
            )
            raise

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a previously generated climate hazard report.

        Args:
            report_id: Report identifier (UUID4 string).

        Returns:
            Report dictionary or None if not found.
        """
        t0 = time.perf_counter()

        try:
            # Check engine first
            if self._compliance_reporter_engine is not None:
                try:
                    engine_result = self._compliance_reporter_engine.get_report(
                        report_id,
                    )
                    if engine_result is not None:
                        elapsed = time.perf_counter() - t0
                        return engine_result if isinstance(engine_result, dict) else None
                except (AttributeError, TypeError, KeyError):
                    pass

            # Fallback to in-memory store
            cached = self._reports.get(report_id)
            elapsed = time.perf_counter() - t0
            return cached.model_dump() if cached is not None else None

        except Exception as exc:
            logger.error("get_report failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Pipeline orchestration (delegate to HazardPipelineEngine)
    # ==================================================================

    def run_pipeline(
        self,
        stages: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the full or partial climate hazard assessment pipeline.

        Delegates to the HazardPipelineEngine for end-to-end pipeline
        orchestration covering ingestion, risk calculation, scenario
        projection, exposure assessment, vulnerability scoring, and
        reporting stages.

        Args:
            stages: Optional list of pipeline stages to execute.
                When None, all 7 stages are executed.
            parameters: Pipeline-wide parameters.
            **kwargs: Additional keyword arguments passed to the engine.

        Returns:
            Dictionary with pipeline execution results.
        """
        t0 = time.perf_counter()

        try:
            # Delegate to engine
            engine_result: Optional[Dict[str, Any]] = None
            if self._hazard_pipeline_engine is not None:
                try:
                    engine_result = self._hazard_pipeline_engine.run_pipeline(
                        stages=stages,
                        parameters=parameters or {},
                        **kwargs,
                    )
                except (AttributeError, TypeError):
                    pass

            # Build response
            pipeline_id = (
                engine_result.get("pipeline_id", _new_uuid())
                if engine_result else _new_uuid()
            )
            stages_completed = (
                engine_result.get("stages_completed", 0)
                if engine_result else 0
            )
            stages_total = (
                engine_result.get("stages_total", 7)
                if engine_result else 7
            )
            overall_status = (
                engine_result.get("overall_status", "pending")
                if engine_result else "pending"
            )
            stage_results = (
                engine_result.get("stage_results", [])
                if engine_result else []
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            response = PipelineResponse(
                pipeline_id=pipeline_id,
                stages_completed=stages_completed,
                stages_total=stages_total,
                stage_results=stage_results,
                overall_status=overall_status,
                duration_ms=elapsed_ms,
            )
            response.provenance_hash = _compute_hash(response)

            # Store in cache
            self._pipeline_results[response.pipeline_id] = response

            # Record provenance
            self.provenance.record(
                entity_type="compliance_report",
                action="run_pipeline",
                entity_id=response.pipeline_id,
                metadata={
                    "stages_completed": stages_completed,
                    "stages_total": stages_total,
                    "overall_status": overall_status,
                    "duration_ms": elapsed_ms,
                },
            )

            # Record metrics
            record_pipeline("full_pipeline", overall_status)
            elapsed = time.perf_counter() - t0
            observe_pipeline_duration("full_pipeline", elapsed)
            self._total_pipeline_runs += 1

            logger.info(
                "Pipeline run %s: stages=%d/%d status=%s "
                "duration=%.2fms",
                response.pipeline_id,
                stages_completed,
                stages_total,
                overall_status,
                elapsed_ms,
            )
            return response.model_dump()

        except Exception as exc:
            logger.error("run_pipeline failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # Health and statistics
    # ==================================================================

    def get_health(self) -> Dict[str, Any]:
        """Perform a health check on the climate hazard connector service.

        Returns a dictionary with health status for each engine and
        the overall service.

        Returns:
            Dictionary with health check results including:
            - ``status``: Overall service status (healthy, degraded,
                unhealthy).
            - ``engines``: Per-engine availability status.
            - ``started``: Whether the service has been started.
            - ``statistics``: Summary statistics.
            - ``provenance_chain_valid``: Whether the provenance chain
                is intact.
            - ``timestamp``: ISO-8601 UTC timestamp of the check.
        """
        t0 = time.perf_counter()

        engines: Dict[str, str] = {
            "hazard_database": (
                "available"
                if self._hazard_database_engine is not None
                else "unavailable"
            ),
            "risk_index": (
                "available"
                if self._risk_index_engine is not None
                else "unavailable"
            ),
            "scenario_projector": (
                "available"
                if self._scenario_projector_engine is not None
                else "unavailable"
            ),
            "exposure_assessor": (
                "available"
                if self._exposure_assessor_engine is not None
                else "unavailable"
            ),
            "vulnerability_scorer": (
                "available"
                if self._vulnerability_scorer_engine is not None
                else "unavailable"
            ),
            "compliance_reporter": (
                "available"
                if self._compliance_reporter_engine is not None
                else "unavailable"
            ),
            "hazard_pipeline": (
                "available"
                if self._hazard_pipeline_engine is not None
                else "unavailable"
            ),
        }

        available_count = sum(
            1 for s in engines.values() if s == "available"
        )
        total_engines = len(engines)

        if available_count == total_engines:
            overall_status = "healthy"
        elif available_count >= 4:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Verify provenance chain
        chain_valid = self.provenance.verify_chain()

        result = {
            "status": overall_status,
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_engines,
            "started": self._started,
            "statistics": {
                "total_sources": len(self._sources),
                "total_hazard_records": len(self._hazard_data),
                "total_risk_indices": len(self._risk_indices),
                "total_projections": self._total_projections,
                "total_assets": len(self._assets),
                "total_exposure_assessments": self._total_exposure_assessments,
                "total_vulnerability_scores": self._total_vulnerability_scores,
                "total_reports": self._total_reports,
                "total_pipeline_runs": self._total_pipeline_runs,
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

        elapsed = time.perf_counter() - t0

        logger.info(
            "Health check: status=%s engines=%d/%d chain_valid=%s",
            overall_status,
            available_count,
            total_engines,
            chain_valid,
        )
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics for the climate hazard service.

        Returns:
            Dictionary with current service statistics.
        """
        return {
            "total_sources": len(self._sources),
            "total_hazard_records": len(self._hazard_data),
            "total_hazard_events": len(self._hazard_events),
            "total_risk_indices": len(self._risk_indices),
            "total_multi_hazards": len(self._multi_hazards),
            "total_comparisons": len(self._comparisons),
            "total_projections": len(self._scenarios),
            "total_assets": len(self._assets),
            "total_exposures": len(self._exposures),
            "total_portfolio_exposures": len(self._portfolio_exposures),
            "total_vulnerabilities": len(self._vulnerabilities),
            "total_reports": len(self._reports),
            "total_pipeline_runs": len(self._pipeline_results),
        }

    # ==================================================================
    # Provenance and metrics access
    # ==================================================================

    def get_provenance(self) -> ProvenanceTracker:
        """Get the provenance tracker instance.

        Returns:
            ProvenanceTracker instance used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics as a dictionary.

        Returns:
            Dictionary of metric names to current values.
        """
        stats = self.get_statistics()
        return {
            **stats,
            "provenance_entries": self.provenance.entry_count,
            "provenance_chain_valid": self.provenance.verify_chain(),
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def startup(self) -> None:
        """Start the climate hazard connector service.

        Safe to call multiple times. Resets Prometheus gauges to baseline
        values on first call.
        """
        if self._started:
            logger.debug(
                "ClimateHazardService already started; skipping"
            )
            return

        logger.info("ClimateHazardService starting up...")
        self._started = True
        set_active_sources(0)
        set_active_assets(0)
        set_high_risk(0)
        logger.info("ClimateHazardService startup complete")

    def shutdown(self) -> None:
        """Shutdown the climate hazard connector service and release resources."""
        if not self._started:
            return

        self._started = False
        set_active_sources(0)
        set_active_assets(0)
        set_high_risk(0)
        logger.info("ClimateHazardService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> ClimateHazardService:
    """Get or create the singleton ClimateHazardService instance.

    Returns:
        The singleton ClimateHazardService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ClimateHazardService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_climate_hazard(
    app: Any,
    config: Optional[ClimateHazardConfig] = None,
) -> ClimateHazardService:
    """Configure the Climate Hazard Connector Service on a FastAPI application.

    Creates the ClimateHazardService, stores it in app.state, mounts
    the climate hazard API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional climate hazard config.

    Returns:
        ClimateHazardService instance.
    """
    global _singleton_instance

    service = ClimateHazardService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.climate_hazard_service = service

    # Mount climate hazard API router
    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Climate hazard API router mounted")
    else:
        logger.warning(
            "Climate hazard router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Climate hazard service configured on app")
    return service


def get_service() -> ClimateHazardService:
    """Get the singleton ClimateHazardService instance.

    Creates a new instance if one does not exist yet. Uses
    double-checked locking for thread safety.

    Returns:
        ClimateHazardService singleton instance.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ClimateHazardService()
    return _singleton_instance


def get_router() -> Any:
    """Get the climate hazard API router.

    Returns the FastAPI APIRouter from the ``api.router`` module.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.climate_hazard.api.router import router
        return router
    except ImportError:
        logger.warning("Climate hazard API router module not available")
        return None


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "ClimateHazardService",
    # Configuration helpers
    "configure_climate_hazard",
    "get_service",
    "get_router",
    # Response models
    "SourceResponse",
    "HazardDataResponse",
    "HazardEventResponse",
    "RiskIndexResponse",
    "MultiHazardResponse",
    "LocationComparisonResponse",
    "ScenarioResponse",
    "AssetResponse",
    "ExposureResponse",
    "PortfolioExposureResponse",
    "VulnerabilityResponse",
    "ReportResponse",
    "PipelineResponse",
    "HealthResponse",
]
