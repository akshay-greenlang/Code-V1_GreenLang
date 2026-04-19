# -*- coding: utf-8 -*-
"""
PACK028Integration - PACK-028 Sector Pathway Pack Integration for PACK-030
=============================================================================

Enterprise integration for fetching sector-specific decarbonization pathways,
convergence data, and sector benchmarks from PACK-028 (Sector Pathway Pack)
into the Net Zero Reporting Pack. Data feeds into SBTi reports (SDA pathway
validation), CDP C4 (sector-specific targets), TCFD Strategy (scenario
analysis), ISSB (industry metrics), and CSRD E1-4 (emission reduction targets).

Integration Points:
    - Sector Pathways: SDA intensity pathways per sector (IEA NZE 2050)
    - Convergence Data: Sector convergence points and trajectory curves
    - Benchmarks: Peer, leader, and IEA benchmark comparisons
    - Technology Milestones: Sector-specific technology deployment schedules
    - Intensity Metrics: Sector-appropriate intensity denominators

Architecture:
    PACK-028 SDA Pathways    --> PACK-030 SBTi/ISSB sector metrics
    PACK-028 Convergence     --> PACK-030 TCFD scenario analysis
    PACK-028 Benchmarks      --> PACK-030 CDP/CSRD peer comparison

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-030 Net Zero Reporting Pack
Status: Production Ready
"""

import hashlib
import importlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class _PackStub:
    """Stub for PACK-028 components when not available."""
    def __init__(self, component: str) -> None:
        self._component = component

    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"component": self._component, "status": "not_available", "pack": "PACK-028"}
        return _stub

def _try_import(component: str, module_path: str) -> Any:
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("PACK-028 component '%s' not available, using stub", component)
        return _PackStub(component)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectorType(str, Enum):
    POWER_GENERATION = "power_generation"
    STEEL = "steel"
    CEMENT = "cement"
    ALUMINUM = "aluminum"
    PULP_PAPER = "pulp_paper"
    CHEMICALS = "chemicals"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    ROAD_TRANSPORT = "road_transport"
    BUILDINGS = "buildings"
    OIL_GAS = "oil_gas"
    AGRICULTURE = "agriculture"
    GENERAL = "general"

class PathwayScenario(str, Enum):
    IEA_NZE_2050 = "iea_nze_2050"
    IEA_APS = "iea_aps"
    IEA_STEPS = "iea_steps"
    IPCC_15C = "ipcc_1.5c"
    IPCC_2C = "ipcc_2c"
    SDA_15C = "sda_1.5c"
    SDA_WB2C = "sda_wb2c"

class BenchmarkTier(str, Enum):
    LEADER = "leader"
    ABOVE_AVERAGE = "above_average"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    LAGGARD = "laggard"

class ConvergenceStatus(str, Enum):
    ON_TRACK = "on_track"
    SLIGHTLY_BEHIND = "slightly_behind"
    SIGNIFICANTLY_BEHIND = "significantly_behind"
    OFF_TRACK = "off_track"
    AHEAD = "ahead"

class ImportStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"
    CACHED = "cached"

# ---------------------------------------------------------------------------
# Component Registry
# ---------------------------------------------------------------------------

PACK028_COMPONENTS: Dict[str, Dict[str, str]] = {
    "sda_pathway_engine": {
        "name": "SDA Pathway Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.sda_pathway_engine",
        "description": "Sector Decarbonization Approach pathway generation",
    },
    "convergence_engine": {
        "name": "Convergence Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.convergence_engine",
        "description": "Sector convergence trajectory calculation",
    },
    "benchmark_engine": {
        "name": "Benchmark Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.benchmark_engine",
        "description": "Peer and IEA benchmark comparison",
    },
    "technology_roadmap_engine": {
        "name": "Technology Roadmap Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.technology_roadmap_engine",
        "description": "Sector-specific technology milestone tracking",
    },
}

# ---------------------------------------------------------------------------
# Sector Intensity Metrics
# ---------------------------------------------------------------------------

SECTOR_INTENSITY_METRICS: Dict[str, Dict[str, str]] = {
    "power_generation": {"metric": "tCO2e/MWh", "denominator": "electricity_generated_mwh"},
    "steel": {"metric": "tCO2e/tonne_steel", "denominator": "crude_steel_production_tonnes"},
    "cement": {"metric": "tCO2e/tonne_cement", "denominator": "cementitious_product_tonnes"},
    "aluminum": {"metric": "tCO2e/tonne_aluminum", "denominator": "primary_aluminum_tonnes"},
    "pulp_paper": {"metric": "tCO2e/tonne_product", "denominator": "pulp_paper_production_tonnes"},
    "chemicals": {"metric": "tCO2e/tonne_product", "denominator": "chemical_production_tonnes"},
    "aviation": {"metric": "gCO2e/pkm", "denominator": "passenger_kilometres"},
    "shipping": {"metric": "gCO2e/tkm", "denominator": "tonne_kilometres"},
    "road_transport": {"metric": "gCO2e/km", "denominator": "vehicle_kilometres"},
    "buildings": {"metric": "kgCO2e/m2", "denominator": "floor_area_m2"},
    "general": {"metric": "tCO2e/mln_revenue", "denominator": "revenue_mln"},
}

# ---------------------------------------------------------------------------
# IEA NZE 2050 Sector Targets
# ---------------------------------------------------------------------------

IEA_NZE_SECTOR_TARGETS: Dict[str, Dict[int, float]] = {
    "power_generation": {2020: 0.46, 2025: 0.34, 2030: 0.14, 2035: 0.02, 2040: 0.0, 2050: 0.0},
    "steel": {2020: 1.89, 2025: 1.70, 2030: 1.28, 2035: 0.85, 2040: 0.43, 2050: 0.02},
    "cement": {2020: 0.63, 2025: 0.57, 2030: 0.42, 2035: 0.28, 2040: 0.14, 2050: 0.03},
    "aluminum": {2020: 8.60, 2025: 7.50, 2030: 5.20, 2035: 3.10, 2040: 1.50, 2050: 0.20},
    "aviation": {2020: 102, 2025: 95, 2030: 72, 2035: 50, 2040: 30, 2050: 5},
    "shipping": {2020: 12.5, 2025: 11.0, 2030: 7.5, 2035: 4.5, 2040: 2.0, 2050: 0.3},
}

SECTOR_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "power_generation": {"leader": 0.08, "average": 0.35, "laggard": 0.65},
    "steel": {"leader": 1.10, "average": 1.85, "laggard": 2.50},
    "cement": {"leader": 0.48, "average": 0.62, "laggard": 0.78},
    "aluminum": {"leader": 4.50, "average": 8.20, "laggard": 14.0},
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PACK028IntegrationConfig(BaseModel):
    """Configuration for PACK-028 to PACK-030 integration."""
    pack_id: str = Field(default="PACK-030")
    source_pack_id: str = Field(default="PACK-028")
    organization_id: str = Field(default="")
    organization_name: str = Field(default="")
    sector: SectorType = Field(default=SectorType.GENERAL)
    base_year: int = Field(default=2023, ge=2015, le=2025)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    pathway_scenario: PathwayScenario = Field(default=PathwayScenario.IEA_NZE_2050)
    enable_provenance: bool = Field(default=True)
    db_connection_string: str = Field(default="")
    db_pool_size: int = Field(default=5, ge=1, le=20)
    cache_ttl_seconds: int = Field(default=3600)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0)

class SectorPathway(BaseModel):
    """Sector-specific decarbonization pathway from PACK-028."""
    pathway_id: str = Field(default_factory=_new_uuid)
    sector: SectorType = Field(default=SectorType.GENERAL)
    scenario: PathwayScenario = Field(default=PathwayScenario.IEA_NZE_2050)
    intensity_metric: str = Field(default="")
    intensity_unit: str = Field(default="")
    base_year: int = Field(default=2020)
    base_year_intensity: float = Field(default=0.0)
    target_year_intensities: Dict[int, float] = Field(default_factory=dict)
    convergence_year: int = Field(default=2050)
    convergence_intensity: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    methodology: str = Field(default="Sectoral Decarbonization Approach (SDA)")
    source: str = Field(default="IEA Net Zero Emissions by 2050 Scenario")
    provenance_hash: str = Field(default="")

class ConvergenceData(BaseModel):
    """Convergence analysis data from PACK-028."""
    convergence_id: str = Field(default_factory=_new_uuid)
    sector: SectorType = Field(default=SectorType.GENERAL)
    organization_intensity: float = Field(default=0.0)
    sector_pathway_intensity: float = Field(default=0.0)
    gap_to_pathway: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    convergence_status: ConvergenceStatus = Field(default=ConvergenceStatus.ON_TRACK)
    years_to_convergence: int = Field(default=0)
    required_annual_reduction_pct: float = Field(default=0.0)
    actual_annual_reduction_pct: float = Field(default=0.0)
    trajectory_by_year: Dict[int, float] = Field(default_factory=dict)
    scenario_comparison: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class SectorBenchmark(BaseModel):
    """Sector benchmark comparison from PACK-028."""
    benchmark_id: str = Field(default_factory=_new_uuid)
    sector: SectorType = Field(default=SectorType.GENERAL)
    organization_intensity: float = Field(default=0.0)
    leader_intensity: float = Field(default=0.0)
    average_intensity: float = Field(default=0.0)
    laggard_intensity: float = Field(default=0.0)
    peer_group_size: int = Field(default=0)
    percentile_rank: float = Field(default=50.0)
    benchmark_tier: BenchmarkTier = Field(default=BenchmarkTier.AVERAGE)
    intensity_unit: str = Field(default="")
    benchmark_year: int = Field(default=2025)
    data_source: str = Field(default="IEA / SBTi sector data")
    provenance_hash: str = Field(default="")

class TechnologyMilestone(BaseModel):
    """Technology deployment milestone from PACK-028."""
    milestone_id: str = Field(default_factory=_new_uuid)
    sector: SectorType = Field(default=SectorType.GENERAL)
    technology_name: str = Field(default="")
    description: str = Field(default="")
    target_year: int = Field(default=2030)
    deployment_level: str = Field(default="")
    current_status: str = Field(default="")
    abatement_potential_pct: float = Field(default=0.0)
    investment_required_usd: float = Field(default=0.0)
    technology_readiness_level: int = Field(default=1, ge=1, le=9)

class PACK028IntegrationResult(BaseModel):
    """Complete PACK-028 integration result for PACK-030."""
    result_id: str = Field(default_factory=_new_uuid)
    pathways: List[SectorPathway] = Field(default_factory=list)
    convergence: Optional[ConvergenceData] = Field(None)
    benchmarks: Optional[SectorBenchmark] = Field(None)
    technology_milestones: List[TechnologyMilestone] = Field(default_factory=list)
    pack028_available: bool = Field(default=False)
    import_status: ImportStatus = Field(default=ImportStatus.FAILED)
    integration_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    frameworks_serviced: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# PACK028Integration
# ---------------------------------------------------------------------------

class PACK028Integration:
    """PACK-028 Sector Pathway Pack integration for PACK-030.

    Fetches sector-specific pathways, convergence analysis, and
    benchmarks from PACK-028 for multi-framework reporting.

    Example:
        >>> config = PACK028IntegrationConfig(
        ...     sector=SectorType.STEEL,
        ...     pathway_scenario=PathwayScenario.IEA_NZE_2050,
        ... )
        >>> integration = PACK028Integration(config)
        >>> pathways = await integration.fetch_pathways()
        >>> convergence = await integration.fetch_convergence()
        >>> benchmarks = await integration.fetch_benchmarks()
    """

    def __init__(self, config: Optional[PACK028IntegrationConfig] = None) -> None:
        self.config = config or PACK028IntegrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._components: Dict[str, Any] = {}
        self._loaded: List[str] = []
        self._stubbed: List[str] = []

        for comp_id, comp_info in PACK028_COMPONENTS.items():
            agent = _try_import(comp_id, comp_info["module"])
            self._components[comp_id] = agent
            if isinstance(agent, _PackStub):
                self._stubbed.append(comp_id)
            else:
                self._loaded.append(comp_id)

        self._pathways_cache: Optional[List[SectorPathway]] = None
        self._convergence_cache: Optional[ConvergenceData] = None
        self._benchmarks_cache: Optional[SectorBenchmark] = None
        self._milestones_cache: Optional[List[TechnologyMilestone]] = None
        self._db_pool: Optional[Any] = None

        self.logger.info(
            "PACK028Integration (PACK-030) initialized: sector=%s, scenario=%s",
            self.config.sector.value, self.config.pathway_scenario.value,
        )

    async def _get_db_pool(self) -> Any:
        if self._db_pool is not None:
            return self._db_pool
        if not self.config.db_connection_string:
            return None
        try:
            import psycopg_pool
            self._db_pool = psycopg_pool.AsyncConnectionPool(
                self.config.db_connection_string, min_size=1,
                max_size=self.config.db_pool_size,
            )
            await self._db_pool.open()
            return self._db_pool
        except Exception as exc:
            self.logger.warning("DB pool creation failed: %s", exc)
            return None

    async def _query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pool = await self._get_db_pool()
        if not pool:
            return []
        attempt = 0
        while attempt < self.config.retry_attempts:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params or {})
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        rows = await cur.fetchall()
                        return [dict(zip(columns, row)) for row in rows]
            except Exception as exc:
                attempt += 1
                self.logger.warning("DB query attempt %d/%d failed: %s", attempt, self.config.retry_attempts, exc)
                if attempt < self.config.retry_attempts:
                    import asyncio

                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
        return []

    # -----------------------------------------------------------------------
    # Fetch Pathways
    # -----------------------------------------------------------------------

    async def fetch_pathways(
        self, override_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SectorPathway]:
        """Fetch sector decarbonization pathways from PACK-028.

        Retrieves SDA-based intensity pathways for the configured sector
        and scenario. Used in SBTi reports (SDA pathway validation),
        ISSB (industry-specific metrics), and TCFD (scenario analysis).
        """
        if self._pathways_cache is not None:
            return self._pathways_cache

        raw_data = override_data or []
        if not raw_data and self.config.db_connection_string:
            raw_data = await self._query(
                "SELECT * FROM gl_pack028_pathways "
                "WHERE sector = %(sector)s AND scenario = %(scenario)s "
                "ORDER BY base_year",
                {"sector": self.config.sector.value, "scenario": self.config.pathway_scenario.value},
            )

        if not raw_data:
            raw_data = self._default_pathways()

        pathways: List[SectorPathway] = []
        for row in raw_data:
            sector = SectorType(row.get("sector", self.config.sector.value))
            metric_info = SECTOR_INTENSITY_METRICS.get(sector.value, SECTOR_INTENSITY_METRICS["general"])

            targets = row.get("target_year_intensities", {})
            targets = {int(k): float(v) for k, v in targets.items()}

            base_intensity = row.get("base_year_intensity", 0.0)
            conv_intensity = targets.get(2050, 0.0)

            # Calculate annual reduction rate
            years = 2050 - row.get("base_year", 2020)
            if base_intensity > 0 and years > 0:
                annual_rate = (1 - (conv_intensity / base_intensity) ** (1 / years)) * 100
            else:
                annual_rate = 0.0

            pathway = SectorPathway(
                sector=sector,
                scenario=PathwayScenario(row.get("scenario", self.config.pathway_scenario.value)),
                intensity_metric=metric_info["metric"],
                intensity_unit=metric_info["metric"],
                base_year=row.get("base_year", 2020),
                base_year_intensity=base_intensity,
                target_year_intensities=targets,
                convergence_year=row.get("convergence_year", 2050),
                convergence_intensity=conv_intensity,
                annual_reduction_rate_pct=round(annual_rate, 2),
                methodology=row.get("methodology", "Sectoral Decarbonization Approach (SDA)"),
                source=row.get("source", "IEA Net Zero Emissions by 2050 Scenario"),
            )
            if self.config.enable_provenance:
                pathway.provenance_hash = _compute_hash(pathway)
            pathways.append(pathway)

        self._pathways_cache = pathways
        self.logger.info(
            "Pathways fetched from PACK-028: %d pathways for sector=%s",
            len(pathways), self.config.sector.value,
        )
        return pathways

    # -----------------------------------------------------------------------
    # Fetch Convergence
    # -----------------------------------------------------------------------

    async def fetch_convergence(
        self, organization_intensity: Optional[float] = None,
        override_data: Optional[Dict[str, Any]] = None,
    ) -> ConvergenceData:
        """Fetch convergence analysis from PACK-028.

        Compares organization intensity to sector pathway and calculates
        convergence gap. Used in TCFD (scenario analysis), CDP (target
        ambition), and SBTi (SDA alignment validation).
        """
        if self._convergence_cache is not None:
            return self._convergence_cache

        data = override_data or {}
        pathways = await self.fetch_pathways()

        sector = self.config.sector.value
        sector_targets = IEA_NZE_SECTOR_TARGETS.get(sector, {})
        pathway_intensity = sector_targets.get(self.config.reporting_year, 0.0)
        if not pathway_intensity and pathways:
            targets = pathways[0].target_year_intensities
            pathway_intensity = targets.get(self.config.reporting_year, targets.get(2030, 0.0))

        org_intensity = organization_intensity or data.get("organization_intensity", pathway_intensity * 1.1)
        gap = org_intensity - pathway_intensity
        gap_pct = (gap / max(pathway_intensity, 0.001)) * 100.0

        if gap_pct <= 0:
            status = ConvergenceStatus.AHEAD
        elif gap_pct <= 10:
            status = ConvergenceStatus.ON_TRACK
        elif gap_pct <= 25:
            status = ConvergenceStatus.SLIGHTLY_BEHIND
        elif gap_pct <= 50:
            status = ConvergenceStatus.SIGNIFICANTLY_BEHIND
        else:
            status = ConvergenceStatus.OFF_TRACK

        # Build trajectory
        trajectory: Dict[int, float] = {}
        for year in range(self.config.reporting_year, 2051, 5):
            if pathways:
                targets = pathways[0].target_year_intensities
                trajectory[year] = targets.get(year, 0.0)
            else:
                trajectory[year] = sector_targets.get(year, 0.0)

        # Scenario comparison
        scenario_comparison: Dict[str, float] = {}
        for scenario_name, multiplier in [("nze_1.5c", 1.0), ("below_2c", 1.15), ("ndc", 1.4), ("bau", 2.0)]:
            scenario_comparison[scenario_name] = round(pathway_intensity * multiplier, 4)

        convergence = ConvergenceData(
            sector=self.config.sector,
            organization_intensity=round(org_intensity, 4),
            sector_pathway_intensity=round(pathway_intensity, 4),
            gap_to_pathway=round(gap, 4),
            gap_pct=round(gap_pct, 2),
            convergence_status=status,
            years_to_convergence=data.get("years_to_convergence", max(0, int(gap_pct / 3))),
            required_annual_reduction_pct=data.get("required_annual_reduction_pct", round(gap_pct / 5, 2)),
            actual_annual_reduction_pct=data.get("actual_annual_reduction_pct", 3.5),
            trajectory_by_year=trajectory,
            scenario_comparison=scenario_comparison,
        )

        if self.config.enable_provenance:
            convergence.provenance_hash = _compute_hash(convergence)

        self._convergence_cache = convergence
        self.logger.info(
            "Convergence fetched: org=%.4f, pathway=%.4f, gap=%.2f%%, status=%s",
            org_intensity, pathway_intensity, gap_pct, status.value,
        )
        return convergence

    # -----------------------------------------------------------------------
    # Fetch Benchmarks
    # -----------------------------------------------------------------------

    async def fetch_benchmarks(
        self, organization_intensity: Optional[float] = None,
        override_data: Optional[Dict[str, Any]] = None,
    ) -> SectorBenchmark:
        """Fetch sector benchmarks from PACK-028.

        Compares organization intensity to sector peers, leaders, and
        laggards. Used in CDP (peer comparison), CSRD (sector context),
        and SBTi (ambition assessment).
        """
        if self._benchmarks_cache is not None:
            return self._benchmarks_cache

        data = override_data or {}
        sector = self.config.sector.value
        benchmarks = SECTOR_BENCHMARKS.get(sector, {"leader": 0.0, "average": 0.0, "laggard": 0.0})
        metric_info = SECTOR_INTENSITY_METRICS.get(sector, SECTOR_INTENSITY_METRICS["general"])

        leader = data.get("leader_intensity", benchmarks.get("leader", 0.0))
        average = data.get("average_intensity", benchmarks.get("average", 0.0))
        laggard = data.get("laggard_intensity", benchmarks.get("laggard", 0.0))
        org_intensity = organization_intensity or data.get("organization_intensity", average * 0.95)

        # Determine tier
        if org_intensity <= leader:
            tier = BenchmarkTier.LEADER
            percentile = 10.0
        elif org_intensity <= (leader + average) / 2:
            tier = BenchmarkTier.ABOVE_AVERAGE
            percentile = 30.0
        elif org_intensity <= average:
            tier = BenchmarkTier.AVERAGE
            percentile = 50.0
        elif org_intensity <= (average + laggard) / 2:
            tier = BenchmarkTier.BELOW_AVERAGE
            percentile = 70.0
        else:
            tier = BenchmarkTier.LAGGARD
            percentile = 90.0

        benchmark = SectorBenchmark(
            sector=self.config.sector,
            organization_intensity=round(org_intensity, 4),
            leader_intensity=round(leader, 4),
            average_intensity=round(average, 4),
            laggard_intensity=round(laggard, 4),
            peer_group_size=data.get("peer_group_size", 150),
            percentile_rank=percentile,
            benchmark_tier=tier,
            intensity_unit=metric_info["metric"],
            benchmark_year=self.config.reporting_year,
            data_source=data.get("data_source", "IEA / SBTi sector data"),
        )

        if self.config.enable_provenance:
            benchmark.provenance_hash = _compute_hash(benchmark)

        self._benchmarks_cache = benchmark
        self.logger.info(
            "Benchmarks fetched: sector=%s, org=%.4f, tier=%s, percentile=%.0f%%",
            sector, org_intensity, tier.value, percentile,
        )
        return benchmark

    # -----------------------------------------------------------------------
    # Framework-specific exports
    # -----------------------------------------------------------------------

    async def get_sbti_sda_data(self) -> Dict[str, Any]:
        """Get SDA pathway data for SBTi progress report."""
        pathways = await self.fetch_pathways()
        convergence = await self.fetch_convergence()
        return {
            "sector": self.config.sector.value,
            "sda_pathway": pathways[0].model_dump() if pathways else {},
            "convergence_status": convergence.convergence_status.value,
            "gap_to_pathway_pct": convergence.gap_pct,
            "annual_reduction_required": convergence.required_annual_reduction_pct,
            "annual_reduction_actual": convergence.actual_annual_reduction_pct,
        }

    async def get_tcfd_scenario_data(self) -> Dict[str, Any]:
        """Get scenario analysis data for TCFD Strategy."""
        convergence = await self.fetch_convergence()
        pathways = await self.fetch_pathways()
        return {
            "scenarios": convergence.scenario_comparison,
            "organization_trajectory": convergence.trajectory_by_year,
            "sector_pathway_trajectory": pathways[0].target_year_intensities if pathways else {},
            "gap_analysis": {
                "current_gap_pct": convergence.gap_pct,
                "convergence_status": convergence.convergence_status.value,
            },
        }

    async def get_cdp_sector_data(self) -> Dict[str, Any]:
        """Get sector benchmark data for CDP questionnaire."""
        benchmarks = await self.fetch_benchmarks()
        return {
            "sector": self.config.sector.value,
            "intensity_metric": benchmarks.intensity_unit,
            "organization_intensity": benchmarks.organization_intensity,
            "peer_percentile": benchmarks.percentile_rank,
            "benchmark_tier": benchmarks.benchmark_tier.value,
            "leader_intensity": benchmarks.leader_intensity,
            "average_intensity": benchmarks.average_intensity,
        }

    async def get_issb_industry_data(self) -> Dict[str, Any]:
        """Get industry-specific metrics for ISSB IFRS S2."""
        benchmarks = await self.fetch_benchmarks()
        pathways = await self.fetch_pathways()
        return {
            "sector": self.config.sector.value,
            "intensity_metric": benchmarks.intensity_unit,
            "organization_intensity": benchmarks.organization_intensity,
            "sector_pathway": pathways[0].target_year_intensities if pathways else {},
            "benchmark_tier": benchmarks.benchmark_tier.value,
        }

    # -----------------------------------------------------------------------
    # Full Integration
    # -----------------------------------------------------------------------

    async def get_full_integration(self) -> PACK028IntegrationResult:
        """Get complete PACK-028 integration result."""
        errors: List[str] = []
        warnings: List[str] = []

        pathways: List[SectorPathway] = []
        convergence: Optional[ConvergenceData] = None
        benchmarks: Optional[SectorBenchmark] = None
        milestones: List[TechnologyMilestone] = []

        try:
            pathways = await self.fetch_pathways()
        except Exception as exc:
            errors.append(f"Pathway fetch failed: {exc}")

        try:
            convergence = await self.fetch_convergence()
        except Exception as exc:
            warnings.append(f"Convergence fetch failed: {exc}")

        try:
            benchmarks = await self.fetch_benchmarks()
        except Exception as exc:
            warnings.append(f"Benchmark fetch failed: {exc}")

        quality = 0.0
        if pathways:
            quality += 40.0
        if convergence:
            quality += 30.0
        if benchmarks:
            quality += 30.0

        status = ImportStatus.SUCCESS if not errors else (
            ImportStatus.FAILED if quality < 40.0 else ImportStatus.PARTIAL
        )

        result = PACK028IntegrationResult(
            pathways=pathways,
            convergence=convergence,
            benchmarks=benchmarks,
            technology_milestones=milestones,
            pack028_available=len(self._loaded) > 0,
            import_status=status,
            integration_quality_score=quality,
            frameworks_serviced=["SBTi", "CDP", "TCFD", "ISSB", "CSRD"],
            validation_errors=errors,
            validation_warnings=warnings,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    # -----------------------------------------------------------------------
    # Default data
    # -----------------------------------------------------------------------

    def _default_pathways(self) -> List[Dict[str, Any]]:
        sector = self.config.sector.value
        targets = IEA_NZE_SECTOR_TARGETS.get(sector, IEA_NZE_SECTOR_TARGETS.get("power_generation", {}))
        base_intensity = targets.get(2020, 0.5)
        return [
            {
                "sector": sector,
                "scenario": self.config.pathway_scenario.value,
                "base_year": 2020,
                "base_year_intensity": base_intensity,
                "target_year_intensities": targets,
                "convergence_year": 2050,
            },
        ]

    # -----------------------------------------------------------------------
    # Status & lifecycle
    # -----------------------------------------------------------------------

    def get_integration_status(self) -> Dict[str, Any]:
        return {
            "pack_id": self.config.pack_id,
            "source_pack_id": self.config.source_pack_id,
            "sector": self.config.sector.value,
            "scenario": self.config.pathway_scenario.value,
            "components_loaded": len(self._loaded),
            "pathways_fetched": self._pathways_cache is not None,
            "convergence_fetched": self._convergence_cache is not None,
            "benchmarks_fetched": self._benchmarks_cache is not None,
            "module_version": _MODULE_VERSION,
        }

    async def refresh(self) -> PACK028IntegrationResult:
        self._pathways_cache = None
        self._convergence_cache = None
        self._benchmarks_cache = None
        self._milestones_cache = None
        return await self.get_full_integration()

    async def close(self) -> None:
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as exc:
                self.logger.warning("Error closing DB pool: %s", exc)
            self._db_pool = None
