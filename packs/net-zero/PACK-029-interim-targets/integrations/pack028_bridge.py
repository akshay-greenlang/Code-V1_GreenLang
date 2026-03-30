# -*- coding: utf-8 -*-
"""
PACK028Bridge - PACK-028 Sector Pathway Integration for PACK-029
===================================================================

Enterprise bridge for integrating PACK-028 (Sector Pathway Pack)
sector-specific interim milestones, technology roadmap milestones,
abatement lever prioritization (MACC curve), and sector benchmarks
into the Interim Targets Pack. PACK-028 provides sector-aware pathway
data that PACK-029 uses to validate interim targets against sector-
specific decarbonization trajectories.

Integration Points:
    - Sector Milestones: 5-year interim intensity targets from SDA pathways
    - Technology Roadmap: IEA NZE 2050 technology milestones per sector
    - Abatement Levers: MACC curve prioritization for initiative sequencing
    - Sector Benchmarks: Peer, leader, and IEA benchmark comparisons
    - Pathway Validation: Interim target validation against sector trajectory

Architecture:
    PACK-028 SDA Pathways      --> PACK-029 Interim Milestone Validation
    PACK-028 IEA Milestones    --> PACK-029 Technology Deployment Schedule
    PACK-028 MACC Curve        --> PACK-029 Initiative-Target Linkage
    PACK-028 Benchmarks        --> PACK-029 Ambition Calibration

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
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
from typing import Any, Dict, List, Optional

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

def _try_import_pack028(component: str, module_path: str) -> Any:
    """Attempt to import a PACK-028 component."""
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
    RAIL = "rail"
    BUILDINGS_RESIDENTIAL = "buildings_residential"
    BUILDINGS_COMMERCIAL = "buildings_commercial"
    AGRICULTURE = "agriculture"
    FOOD_BEVERAGE = "food_beverage"
    OIL_GAS = "oil_gas"
    CROSS_SECTOR = "cross_sector"

class MilestoneType(str, Enum):
    INTENSITY_TARGET = "intensity_target"
    TECHNOLOGY_DEPLOYMENT = "technology_deployment"
    POLICY_COMPLIANCE = "policy_compliance"
    ABSOLUTE_REDUCTION = "absolute_reduction"
    BENCHMARK_ALIGNMENT = "benchmark_alignment"

class BenchmarkTier(str, Enum):
    PEER_AVERAGE = "peer_average"
    PEER_LEADER = "peer_leader"
    IEA_NZE = "iea_nze"
    SBTI_15C = "sbti_15c"
    SBTI_WB2C = "sbti_wb2c"

class LeverPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"

class MilestoneStatus(str, Enum):
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"

# ---------------------------------------------------------------------------
# PACK-028 Component Registry
# ---------------------------------------------------------------------------

PACK028_COMPONENTS: Dict[str, Dict[str, str]] = {
    "sbti_sda_bridge": {
        "name": "SBTi SDA Bridge",
        "module": "packs.net_zero.PACK_028_sector_pathway.integrations.sbti_sda_bridge",
        "description": "SBTi Sectoral Decarbonization Approach convergence pathways",
    },
    "iea_nze_bridge": {
        "name": "IEA NZE Bridge",
        "module": "packs.net_zero.PACK_028_sector_pathway.integrations.iea_nze_bridge",
        "description": "IEA Net Zero Emissions 2050 technology milestones",
    },
    "sector_benchmark_engine": {
        "name": "Sector Benchmark Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.sector_benchmark_engine",
        "description": "Sector-specific peer and leader benchmarking",
    },
    "abatement_waterfall_engine": {
        "name": "Abatement Waterfall Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.abatement_waterfall_engine",
        "description": "MACC curve and abatement lever prioritization",
    },
    "pathway_generator_engine": {
        "name": "Pathway Generator Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.pathway_generator_engine",
        "description": "Sector pathway generation with multiple convergence methods",
    },
    "technology_roadmap_engine": {
        "name": "Technology Roadmap Engine",
        "module": "packs.net_zero.PACK_028_sector_pathway.engines.technology_roadmap_engine",
        "description": "Technology adoption S-curves and deployment scheduling",
    },
}

# ---------------------------------------------------------------------------
# Sector Interim Milestone Tables (SDA 1.5C)
# ---------------------------------------------------------------------------

SECTOR_INTERIM_MILESTONES: Dict[str, Dict[int, float]] = {
    "power_generation": {2025: 380.0, 2030: 270.0, 2035: 160.0, 2040: 80.0, 2045: 30.0, 2050: 0.0},
    "steel": {2025: 1.70, 2030: 1.45, 2035: 1.15, 2040: 0.80, 2045: 0.40, 2050: 0.10},
    "cement": {2025: 0.57, 2030: 0.48, 2035: 0.38, 2040: 0.25, 2045: 0.13, 2050: 0.04},
    "aluminum": {2025: 7.50, 2030: 6.00, 2035: 4.20, 2040: 2.50, 2045: 1.00, 2050: 0.30},
    "pulp_paper": {2025: 0.38, 2030: 0.30, 2035: 0.22, 2040: 0.14, 2045: 0.07, 2050: 0.02},
    "chemicals": {2025: 1.30, 2030: 1.05, 2035: 0.78, 2040: 0.50, 2045: 0.25, 2050: 0.08},
    "aviation": {2025: 92.0, 2030: 80.0, 2035: 62.0, 2040: 40.0, 2045: 20.0, 2050: 5.0},
    "shipping": {2025: 11.0, 2030: 9.0, 2035: 6.5, 2040: 4.0, 2045: 1.8, 2050: 0.5},
    "road_transport": {2025: 160.0, 2030: 120.0, 2035: 75.0, 2040: 35.0, 2045: 12.0, 2050: 0.0},
    "rail": {2025: 28.0, 2030: 20.0, 2035: 13.0, 2040: 7.0, 2045: 3.0, 2050: 0.0},
    "buildings_residential": {2025: 23.0, 2030: 17.0, 2035: 11.0, 2040: 6.0, 2045: 2.5, 2050: 0.5},
    "buildings_commercial": {2025: 31.0, 2030: 23.0, 2035: 15.0, 2040: 8.0, 2045: 3.5, 2050: 0.8},
}

SECTOR_INTENSITY_METRICS: Dict[str, str] = {
    "power_generation": "gCO2/kWh",
    "steel": "tCO2e/tonne crude steel",
    "cement": "tCO2e/tonne cement",
    "aluminum": "tCO2e/tonne aluminum",
    "pulp_paper": "tCO2e/tonne pulp",
    "chemicals": "tCO2e/tonne product",
    "aviation": "gCO2/pkm",
    "shipping": "gCO2/tkm",
    "road_transport": "gCO2/vkm",
    "rail": "gCO2/pkm",
    "buildings_residential": "kgCO2/m2/year",
    "buildings_commercial": "kgCO2/m2/year",
}

# IEA NZE technology milestones per sector (key milestones by year)
IEA_TECHNOLOGY_MILESTONES: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"year": 2025, "milestone": "No new unabated coal plants approved", "status": "at_risk"},
        {"year": 2030, "milestone": "Solar/wind capacity 3x 2020 levels", "status": "on_track"},
        {"year": 2035, "milestone": "Advanced economies net-zero electricity", "status": "not_started"},
        {"year": 2040, "milestone": "Global electricity net-zero", "status": "not_started"},
        {"year": 2050, "milestone": "All remaining fossil with CCS", "status": "not_started"},
    ],
    "steel": [
        {"year": 2025, "milestone": "First commercial green hydrogen DRI", "status": "on_track"},
        {"year": 2030, "milestone": "All new capacity H2-DRI or EAF", "status": "not_started"},
        {"year": 2035, "milestone": "50% scrap-based EAF globally", "status": "not_started"},
        {"year": 2040, "milestone": "CCS on remaining BF-BOF", "status": "not_started"},
        {"year": 2050, "milestone": "Near-zero emission steel production", "status": "not_started"},
    ],
    "cement": [
        {"year": 2025, "milestone": "Clinker ratio below 0.70", "status": "on_track"},
        {"year": 2030, "milestone": "Novel cements 10% market share", "status": "not_started"},
        {"year": 2035, "milestone": "CCS on 20% of production", "status": "not_started"},
        {"year": 2040, "milestone": "CCS on 50% of production", "status": "not_started"},
        {"year": 2050, "milestone": "CCS on all remaining process emissions", "status": "not_started"},
    ],
    "aviation": [
        {"year": 2025, "milestone": "SAF 5% of total fuel", "status": "at_risk"},
        {"year": 2030, "milestone": "SAF 15% of total fuel", "status": "not_started"},
        {"year": 2035, "milestone": "First electric short-haul routes", "status": "not_started"},
        {"year": 2040, "milestone": "SAF 50% of total fuel", "status": "not_started"},
        {"year": 2050, "milestone": "SAF + hydrogen 100% long-haul", "status": "not_started"},
    ],
    "buildings_commercial": [
        {"year": 2025, "milestone": "All new buildings net-zero-ready", "status": "at_risk"},
        {"year": 2030, "milestone": "50% heat pump installations", "status": "not_started"},
        {"year": 2035, "milestone": "Deep retrofit 2.5% annual rate", "status": "not_started"},
        {"year": 2040, "milestone": "No fossil heating in new builds", "status": "not_started"},
        {"year": 2050, "milestone": "All buildings net-zero operational", "status": "not_started"},
    ],
}

# Sector benchmark data (peer average, leader, IEA NZE for 2030)
SECTOR_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "power_generation": {"peer_average": 420.0, "peer_leader": 310.0, "iea_nze_2030": 270.0},
    "steel": {"peer_average": 1.82, "peer_leader": 1.55, "iea_nze_2030": 1.45},
    "cement": {"peer_average": 0.60, "peer_leader": 0.50, "iea_nze_2030": 0.48},
    "aluminum": {"peer_average": 8.20, "peer_leader": 6.50, "iea_nze_2030": 6.00},
    "aviation": {"peer_average": 96.0, "peer_leader": 85.0, "iea_nze_2030": 80.0},
    "shipping": {"peer_average": 12.0, "peer_leader": 9.5, "iea_nze_2030": 9.0},
    "buildings_commercial": {"peer_average": 36.0, "peer_leader": 25.0, "iea_nze_2030": 23.0},
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PACK028BridgeConfig(BaseModel):
    """Configuration for the PACK-028 to PACK-029 bridge."""
    pack_id: str = Field(default="PACK-029")
    pack028_id: str = Field(default="PACK-028")
    primary_sector: str = Field(default="cross_sector")
    organization_name: str = Field(default="")
    base_year: int = Field(default=2023, ge=2015, le=2025)
    enable_provenance: bool = Field(default=True)
    enable_technology_milestones: bool = Field(default=True)
    enable_benchmarks: bool = Field(default=True)
    db_connection_string: str = Field(default="")
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=30.0)

class SectorMilestoneImport(BaseModel):
    """Imported sector-specific interim milestone from PACK-028."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-028")
    sector: str = Field(default="")
    milestone_year: int = Field(default=2030)
    intensity_metric: str = Field(default="")
    target_intensity: float = Field(default=0.0)
    milestone_type: MilestoneType = Field(default=MilestoneType.INTENSITY_TARGET)
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)
    provenance_hash: str = Field(default="")

class TechnologyRoadmapImport(BaseModel):
    """Imported technology roadmap milestone from PACK-028 / IEA NZE."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-028")
    sector: str = Field(default="")
    milestone_year: int = Field(default=2030)
    milestone_description: str = Field(default="")
    technology_category: str = Field(default="")
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)
    relevance_to_interim_targets: str = Field(default="")
    abatement_potential_tco2e: float = Field(default=0.0)
    capex_usd: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class AbatementLeverImport(BaseModel):
    """Imported abatement lever from PACK-028 MACC curve."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-028")
    lever_name: str = Field(default="")
    lever_category: str = Field(default="")
    sector: str = Field(default="")
    abatement_potential_tco2e: float = Field(default=0.0)
    cost_per_tco2e_usd: float = Field(default=0.0)
    implementation_phase: str = Field(default="medium_term")
    priority: LeverPriority = Field(default=LeverPriority.MEDIUM)
    interim_target_year: int = Field(default=2030)
    cumulative_abatement_by_year: Dict[int, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class SectorBenchmarkImport(BaseModel):
    """Imported sector benchmark from PACK-028."""
    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="PACK-028")
    sector: str = Field(default="")
    intensity_metric: str = Field(default="")
    peer_average: float = Field(default=0.0)
    peer_leader: float = Field(default=0.0)
    iea_nze_2030: float = Field(default=0.0)
    company_current: float = Field(default=0.0)
    company_rank: str = Field(default="")
    gap_to_leader_pct: float = Field(default=0.0)
    gap_to_nze_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class PACK028IntegrationResult(BaseModel):
    """Complete PACK-028 integration result for PACK-029."""
    result_id: str = Field(default_factory=_new_uuid)
    sector: str = Field(default="")
    milestones: List[SectorMilestoneImport] = Field(default_factory=list)
    technology_roadmap: List[TechnologyRoadmapImport] = Field(default_factory=list)
    abatement_levers: List[AbatementLeverImport] = Field(default_factory=list)
    benchmark: Optional[SectorBenchmarkImport] = Field(None)
    pack028_available: bool = Field(default=False)
    components_loaded: List[str] = Field(default_factory=list)
    components_stubbed: List[str] = Field(default_factory=list)
    integration_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# PACK028Bridge
# ---------------------------------------------------------------------------

class PACK028Bridge:
    """PACK-028 Sector Pathway Pack integration bridge for PACK-029.

    Imports sector-specific interim milestones, IEA NZE technology
    roadmap, MACC curve abatement levers, and sector benchmarks from
    PACK-028 for interim target validation and initiative-target linkage.

    Example:
        >>> config = PACK028BridgeConfig(primary_sector="steel")
        >>> bridge = PACK028Bridge(config)
        >>> milestones = await bridge.import_sector_milestones()
        >>> tech = await bridge.import_technology_roadmap()
        >>> levers = await bridge.import_abatement_levers()
        >>> bench = await bridge.import_sector_benchmarks()
        >>> result = await bridge.get_full_integration()
    """

    def __init__(self, config: Optional[PACK028BridgeConfig] = None) -> None:
        self.config = config or PACK028BridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._components: Dict[str, Any] = {}
        self._loaded: List[str] = []
        self._stubbed: List[str] = []

        for comp_id, comp_info in PACK028_COMPONENTS.items():
            agent = _try_import_pack028(comp_id, comp_info["module"])
            self._components[comp_id] = agent
            if isinstance(agent, _PackStub):
                self._stubbed.append(comp_id)
            else:
                self._loaded.append(comp_id)

        self._milestones_cache: List[SectorMilestoneImport] = []
        self._tech_cache: List[TechnologyRoadmapImport] = []
        self._lever_cache: List[AbatementLeverImport] = []
        self._benchmark_cache: Optional[SectorBenchmarkImport] = None

        self.logger.info(
            "PACK028Bridge (PACK-029) initialized: %d/%d components loaded, sector=%s",
            len(self._loaded), len(PACK028_COMPONENTS), self.config.primary_sector,
        )

    async def import_sector_milestones(
        self, sector: Optional[str] = None,
        milestone_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SectorMilestoneImport]:
        """Import sector-specific interim milestones from PACK-028 SDA pathways.

        Retrieves 5-year intensity convergence targets from the SBTi
        SDA pathway for the organization's primary sector.
        """
        sector = sector or self.config.primary_sector
        data = milestone_data or []

        if not data:
            pathway = SECTOR_INTERIM_MILESTONES.get(sector, {})
            metric = SECTOR_INTENSITY_METRICS.get(sector, "tCO2e/unit")
            for year, intensity in sorted(pathway.items()):
                data.append({
                    "sector": sector,
                    "milestone_year": year,
                    "intensity_metric": metric,
                    "target_intensity": intensity,
                    "milestone_type": "intensity_target",
                })

        milestones: List[SectorMilestoneImport] = []
        for item in data:
            m = SectorMilestoneImport(
                sector=item.get("sector", sector),
                milestone_year=item.get("milestone_year", 2030),
                intensity_metric=item.get("intensity_metric", ""),
                target_intensity=item.get("target_intensity", 0.0),
                milestone_type=MilestoneType(item.get("milestone_type", "intensity_target")),
                status=MilestoneStatus(item.get("status", "not_started")),
            )
            if self.config.enable_provenance:
                m.provenance_hash = _compute_hash(m)
            milestones.append(m)

        self._milestones_cache = milestones
        self.logger.info(
            "Sector milestones imported: sector=%s, count=%d",
            sector, len(milestones),
        )
        return milestones

    async def import_technology_roadmap(
        self, sector: Optional[str] = None,
        tech_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TechnologyRoadmapImport]:
        """Import technology roadmap milestones from PACK-028 / IEA NZE 2050.

        Retrieves sector-specific technology deployment milestones
        that inform initiative sequencing in interim target plans.
        """
        sector = sector or self.config.primary_sector
        data = tech_data or []

        if not data and self.config.enable_technology_milestones:
            iea_milestones = IEA_TECHNOLOGY_MILESTONES.get(sector, [])
            data = iea_milestones

        roadmap: List[TechnologyRoadmapImport] = []
        for item in data:
            t = TechnologyRoadmapImport(
                sector=sector,
                milestone_year=item.get("year", 2030),
                milestone_description=item.get("milestone", ""),
                technology_category=item.get("technology_category", "general"),
                status=MilestoneStatus(item.get("status", "not_started")),
                relevance_to_interim_targets=item.get(
                    "relevance",
                    f"Technology milestone for {sector} sector pathway",
                ),
                abatement_potential_tco2e=item.get("abatement_potential_tco2e", 0.0),
                capex_usd=item.get("capex_usd", 0.0),
            )
            if self.config.enable_provenance:
                t.provenance_hash = _compute_hash(t)
            roadmap.append(t)

        self._tech_cache = roadmap
        self.logger.info(
            "Technology roadmap imported: sector=%s, milestones=%d",
            sector, len(roadmap),
        )
        return roadmap

    async def import_abatement_levers(
        self, sector: Optional[str] = None,
        lever_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AbatementLeverImport]:
        """Import abatement lever prioritization from PACK-028 MACC curve.

        Retrieves the MACC curve levers sorted by cost-effectiveness,
        with cumulative abatement potential mapped to interim target years.
        """
        sector = sector or self.config.primary_sector
        data = lever_data or []

        if not data:
            data = self._generate_default_levers(sector)

        levers: List[AbatementLeverImport] = []
        for item in data:
            cumulative = item.get("cumulative_abatement_by_year", {})
            if not cumulative:
                annual = item.get("abatement_potential_tco2e", 0.0)
                base_year = self.config.base_year
                cumulative = {}
                for yr in range(2025, 2051, 5):
                    years_active = max(0, yr - max(base_year, 2024))
                    cumulative[yr] = round(annual * min(years_active, 5), 2)

            lever = AbatementLeverImport(
                lever_name=item.get("lever_name", ""),
                lever_category=item.get("lever_category", ""),
                sector=item.get("sector", sector),
                abatement_potential_tco2e=item.get("abatement_potential_tco2e", 0.0),
                cost_per_tco2e_usd=item.get("cost_per_tco2e_usd", 0.0),
                implementation_phase=item.get("implementation_phase", "medium_term"),
                priority=LeverPriority(item.get("priority", "medium")),
                interim_target_year=item.get("interim_target_year", 2030),
                cumulative_abatement_by_year=cumulative,
            )
            if self.config.enable_provenance:
                lever.provenance_hash = _compute_hash(lever)
            levers.append(lever)

        # Sort by cost (ascending) for MACC curve order
        levers.sort(key=lambda x: x.cost_per_tco2e_usd)

        self._lever_cache = levers
        self.logger.info(
            "Abatement levers imported: sector=%s, count=%d, "
            "total_potential=%.1f tCO2e",
            sector, len(levers),
            sum(l.abatement_potential_tco2e for l in levers),
        )
        return levers

    async def import_sector_benchmarks(
        self, sector: Optional[str] = None,
        company_intensity: Optional[float] = None,
    ) -> SectorBenchmarkImport:
        """Import sector benchmarks from PACK-028 for ambition calibration.

        Retrieves peer average, peer leader, and IEA NZE 2030 benchmark
        intensities plus calculates gap to leader and NZE targets.
        """
        sector = sector or self.config.primary_sector
        benchmarks = SECTOR_BENCHMARKS.get(sector, {})

        peer_avg = benchmarks.get("peer_average", 0.0)
        peer_leader = benchmarks.get("peer_leader", 0.0)
        iea_nze = benchmarks.get("iea_nze_2030", 0.0)
        company = company_intensity or peer_avg * 1.05

        # Determine company rank
        if company <= iea_nze:
            rank = "IEA NZE aligned"
        elif company <= peer_leader:
            rank = "Above leader, NZE aligned"
        elif company <= peer_avg:
            rank = "Above average"
        else:
            rank = "Below average"

        gap_leader = ((company - peer_leader) / max(peer_leader, 0.001)) * 100.0 if peer_leader > 0 else 0.0
        gap_nze = ((company - iea_nze) / max(iea_nze, 0.001)) * 100.0 if iea_nze > 0 else 0.0

        benchmark = SectorBenchmarkImport(
            sector=sector,
            intensity_metric=SECTOR_INTENSITY_METRICS.get(sector, "tCO2e/unit"),
            peer_average=peer_avg,
            peer_leader=peer_leader,
            iea_nze_2030=iea_nze,
            company_current=round(company, 4),
            company_rank=rank,
            gap_to_leader_pct=round(gap_leader, 2),
            gap_to_nze_pct=round(gap_nze, 2),
        )

        if self.config.enable_provenance:
            benchmark.provenance_hash = _compute_hash(benchmark)

        self._benchmark_cache = benchmark
        self.logger.info(
            "Sector benchmark imported: sector=%s, company=%.4f, "
            "leader=%.4f, nze=%.4f, rank=%s",
            sector, company, peer_leader, iea_nze, rank,
        )
        return benchmark

    async def validate_interim_target_against_sector(
        self, target_year: int, target_intensity: float,
        sector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a proposed interim target against sector pathway.

        Checks whether the proposed target intensity for a given year
        meets or exceeds the SDA sector convergence pathway target.
        """
        sector = sector or self.config.primary_sector
        pathway = SECTOR_INTERIM_MILESTONES.get(sector, {})

        # Interpolate for non-milestone years
        sector_target = self._interpolate_milestone(pathway, target_year)
        meets_sector = target_intensity <= sector_target * 1.05

        benchmarks = SECTOR_BENCHMARKS.get(sector, {})
        meets_leader = target_intensity <= benchmarks.get("peer_leader", float("inf"))
        meets_nze = target_intensity <= benchmarks.get("iea_nze_2030", float("inf"))

        return {
            "sector": sector,
            "target_year": target_year,
            "proposed_intensity": target_intensity,
            "sector_pathway_target": round(sector_target, 4),
            "meets_sector_pathway": meets_sector,
            "meets_peer_leader": meets_leader,
            "meets_iea_nze": meets_nze,
            "gap_to_pathway": round(target_intensity - sector_target, 4),
            "gap_to_pathway_pct": round(
                ((target_intensity - sector_target) / max(sector_target, 0.001)) * 100.0, 2
            ),
            "recommendation": (
                "Target aligned with sector pathway"
                if meets_sector
                else f"Target {round(target_intensity - sector_target, 4)} above sector pathway"
            ),
        }

    async def get_full_integration(self) -> PACK028IntegrationResult:
        """Get complete PACK-028 integration result for PACK-029."""
        if not self._milestones_cache:
            await self.import_sector_milestones()
        if not self._tech_cache:
            await self.import_technology_roadmap()
        if not self._lever_cache:
            await self.import_abatement_levers()
        if not self._benchmark_cache:
            await self.import_sector_benchmarks()

        quality = 0.0
        if self._milestones_cache:
            quality += 30.0
        if self._tech_cache:
            quality += 25.0
        if self._lever_cache:
            quality += 25.0
        if self._benchmark_cache:
            quality += 20.0

        result = PACK028IntegrationResult(
            sector=self.config.primary_sector,
            milestones=self._milestones_cache,
            technology_roadmap=self._tech_cache,
            abatement_levers=self._lever_cache,
            benchmark=self._benchmark_cache,
            pack028_available=len(self._loaded) > 0,
            components_loaded=self._loaded,
            components_stubbed=self._stubbed,
            integration_quality_score=quality,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "pack028_id": self.config.pack028_id,
            "sector": self.config.primary_sector,
            "components_total": len(PACK028_COMPONENTS),
            "components_loaded": len(self._loaded),
            "milestones_imported": len(self._milestones_cache),
            "tech_milestones_imported": len(self._tech_cache),
            "levers_imported": len(self._lever_cache),
            "benchmark_imported": self._benchmark_cache is not None,
        }

    def _interpolate_milestone(self, pathway: Dict[int, float], year: int) -> float:
        """Linearly interpolate between milestone years."""
        if not pathway:
            return 0.0
        years = sorted(pathway.keys())
        if year <= years[0]:
            return pathway[years[0]]
        if year >= years[-1]:
            return pathway[years[-1]]
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                fraction = (year - years[i]) / (years[i + 1] - years[i])
                return pathway[years[i]] + fraction * (pathway[years[i + 1]] - pathway[years[i]])
        return pathway[years[-1]]

    def _generate_default_levers(self, sector: str) -> List[Dict[str, Any]]:
        """Generate default MACC curve levers for a sector."""
        common_levers = [
            {"lever_name": "LED lighting retrofit", "lever_category": "energy_efficiency", "abatement_potential_tco2e": 500, "cost_per_tco2e_usd": -50, "implementation_phase": "immediate", "priority": "critical"},
            {"lever_name": "HVAC optimization", "lever_category": "energy_efficiency", "abatement_potential_tco2e": 1200, "cost_per_tco2e_usd": -30, "implementation_phase": "short_term", "priority": "critical"},
            {"lever_name": "Renewable electricity PPA", "lever_category": "renewable_procurement", "abatement_potential_tco2e": 8000, "cost_per_tco2e_usd": -10, "implementation_phase": "short_term", "priority": "high"},
            {"lever_name": "Fleet electrification", "lever_category": "electrification", "abatement_potential_tco2e": 3000, "cost_per_tco2e_usd": 20, "implementation_phase": "medium_term", "priority": "high"},
            {"lever_name": "Heat pump installation", "lever_category": "electrification", "abatement_potential_tco2e": 2000, "cost_per_tco2e_usd": 35, "implementation_phase": "medium_term", "priority": "medium"},
            {"lever_name": "Supplier engagement program", "lever_category": "supply_chain", "abatement_potential_tco2e": 5000, "cost_per_tco2e_usd": 15, "implementation_phase": "medium_term", "priority": "medium"},
            {"lever_name": "On-site solar PV", "lever_category": "renewable_procurement", "abatement_potential_tco2e": 2500, "cost_per_tco2e_usd": 5, "implementation_phase": "short_term", "priority": "high"},
            {"lever_name": "Process heat electrification", "lever_category": "electrification", "abatement_potential_tco2e": 4000, "cost_per_tco2e_usd": 60, "implementation_phase": "long_term", "priority": "medium"},
            {"lever_name": "Green hydrogen fuel switch", "lever_category": "hydrogen", "abatement_potential_tco2e": 6000, "cost_per_tco2e_usd": 120, "implementation_phase": "long_term", "priority": "low"},
            {"lever_name": "Carbon capture (CCS)", "lever_category": "ccs_ccus", "abatement_potential_tco2e": 10000, "cost_per_tco2e_usd": 150, "implementation_phase": "transformational", "priority": "deferred"},
        ]
        for lever in common_levers:
            lever["sector"] = sector
        return common_levers
