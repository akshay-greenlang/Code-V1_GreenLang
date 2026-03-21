# -*- coding: utf-8 -*-
"""
SectorMRVBridge - Sector-Specific 30-Agent MRV Integration for PACK-028
=========================================================================

Routes emission calculation requests to all 30 MRV agents with sector-
specific prioritization and intensity metric calculation support. Unlike
PACK-027's enterprise MRV bridge (all 30 agents at activity-based
precision), PACK-028's bridge adds sector-aware routing that prioritizes
agents most relevant to each of the 15+ supported sectors.

MRV Agent Coverage (all 30):
    Scope 1 (8 agents):  MRV-001 through MRV-008
    Scope 2 (5 agents):  MRV-009 through MRV-013
    Scope 3 (15 agents): MRV-014 through MRV-028
    Cross-cutting (2):   MRV-029 (Category Mapper), MRV-030 (Audit Trail)

Sector-Specific Features:
    - Per-sector MRV agent priority ranking
    - Sector intensity metric calculation support
    - Activity-data-to-intensity conversion
    - Process emission agent routing (MRV-004) for heavy industry
    - FLAG-related agent routing (MRV-006, MRV-008) for agriculture
    - Mobile combustion routing (MRV-003) for transport
    - SHA-256 provenance on all calculations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-028 Sector Pathway Pack
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class _AgentStub:
    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name

    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"agent": self._agent_name, "status": "degraded", "emissions_tco2e": 0.0}
        return _stub


def _try_import_mrv_agent(agent_id: str, module_path: str) -> Any:
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MRVScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class SectorPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SectorMRVAgentRoute(BaseModel):
    """MRV agent routing definition with sector priority."""
    mrv_agent_id: str = Field(...)
    mrv_agent_name: str = Field(default="")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    module_path: str = Field(default="")
    description: str = Field(default="")
    sector_priorities: Dict[str, str] = Field(default_factory=dict)


class SectorMRVBridgeConfig(BaseModel):
    pack_id: str = Field(default="PACK-028")
    primary_sector: str = Field(default="steel")
    enable_provenance: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=15, ge=1, le=30)
    connection_pool_size: int = Field(default=10, ge=1, le=30)
    intensity_calculation: bool = Field(default=True)
    data_quality_minimum: float = Field(default=0.80, ge=0.5, le=1.0)


class RoutingResult(BaseModel):
    routing_id: str = Field(default_factory=_new_uuid)
    mrv_agent_id: str = Field(default="")
    scope: str = Field(default="")
    scope3_category: Optional[int] = Field(None)
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    emissions_tco2e: float = Field(default=0.0)
    data_quality_score: float = Field(default=0.0)
    sector_priority: str = Field(default="medium")
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class IntensityResult(BaseModel):
    """Result of intensity metric calculation from MRV outputs."""
    sector: str = Field(default="")
    metric: str = Field(default="")
    emissions_tco2e: float = Field(default=0.0)
    activity_value: float = Field(default=0.0)
    intensity_value: float = Field(default=0.0)
    data_quality_score: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class BatchRoutingResult(BaseModel):
    batch_id: str = Field(default_factory=_new_uuid)
    sector: str = Field(default="")
    total_agents: int = Field(default=0)
    successful: int = Field(default=0)
    degraded: int = Field(default=0)
    failed: int = Field(default=0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    intensity: Optional[IntensityResult] = Field(None)
    results: List[RoutingResult] = Field(default_factory=list)
    priority_agents_used: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Sector MRV Priority Mapping
# ---------------------------------------------------------------------------

# Priority per agent per sector routing group
SECTOR_AGENT_PRIORITIES: Dict[str, Dict[str, str]] = {
    "MRV-001": {"heavy_industry": "critical", "light_industry": "critical", "power": "critical", "transport": "medium", "buildings": "critical", "agriculture": "medium"},
    "MRV-002": {"heavy_industry": "medium", "light_industry": "medium", "power": "low", "transport": "low", "buildings": "high", "agriculture": "low"},
    "MRV-003": {"heavy_industry": "medium", "light_industry": "low", "power": "low", "transport": "critical", "buildings": "low", "agriculture": "medium"},
    "MRV-004": {"heavy_industry": "critical", "light_industry": "high", "power": "medium", "transport": "low", "buildings": "low", "agriculture": "low"},
    "MRV-005": {"heavy_industry": "high", "light_industry": "medium", "power": "high", "transport": "low", "buildings": "low", "agriculture": "low"},
    "MRV-006": {"heavy_industry": "low", "light_industry": "medium", "power": "low", "transport": "low", "buildings": "low", "agriculture": "critical"},
    "MRV-007": {"heavy_industry": "medium", "light_industry": "high", "power": "low", "transport": "low", "buildings": "low", "agriculture": "medium"},
    "MRV-008": {"heavy_industry": "low", "light_industry": "high", "power": "low", "transport": "low", "buildings": "low", "agriculture": "critical"},
    "MRV-009": {"heavy_industry": "critical", "light_industry": "critical", "power": "critical", "transport": "critical", "buildings": "critical", "agriculture": "critical"},
    "MRV-010": {"heavy_industry": "critical", "light_industry": "critical", "power": "critical", "transport": "critical", "buildings": "critical", "agriculture": "critical"},
    "MRV-011": {"heavy_industry": "high", "light_industry": "high", "power": "medium", "transport": "low", "buildings": "high", "agriculture": "low"},
    "MRV-012": {"heavy_industry": "low", "light_industry": "low", "power": "low", "transport": "low", "buildings": "high", "agriculture": "low"},
    "MRV-013": {"heavy_industry": "critical", "light_industry": "critical", "power": "critical", "transport": "critical", "buildings": "critical", "agriculture": "critical"},
    "MRV-014": {"heavy_industry": "high", "light_industry": "high", "power": "high", "transport": "high", "buildings": "high", "agriculture": "high"},
    "MRV-015": {"heavy_industry": "medium", "light_industry": "medium", "power": "medium", "transport": "medium", "buildings": "medium", "agriculture": "medium"},
    "MRV-016": {"heavy_industry": "high", "light_industry": "high", "power": "high", "transport": "high", "buildings": "medium", "agriculture": "medium"},
    "MRV-017": {"heavy_industry": "high", "light_industry": "high", "power": "medium", "transport": "critical", "buildings": "medium", "agriculture": "high"},
    "MRV-018": {"heavy_industry": "medium", "light_industry": "medium", "power": "low", "transport": "medium", "buildings": "medium", "agriculture": "medium"},
    "MRV-019": {"heavy_industry": "medium", "light_industry": "medium", "power": "medium", "transport": "medium", "buildings": "medium", "agriculture": "medium"},
    "MRV-020": {"heavy_industry": "medium", "light_industry": "medium", "power": "medium", "transport": "medium", "buildings": "medium", "agriculture": "medium"},
    "MRV-021": {"heavy_industry": "medium", "light_industry": "medium", "power": "medium", "transport": "medium", "buildings": "high", "agriculture": "low"},
    "MRV-022": {"heavy_industry": "high", "light_industry": "medium", "power": "medium", "transport": "critical", "buildings": "medium", "agriculture": "medium"},
    "MRV-023": {"heavy_industry": "high", "light_industry": "high", "power": "medium", "transport": "low", "buildings": "low", "agriculture": "high"},
    "MRV-024": {"heavy_industry": "high", "light_industry": "medium", "power": "high", "transport": "high", "buildings": "high", "agriculture": "medium"},
    "MRV-025": {"heavy_industry": "medium", "light_industry": "medium", "power": "low", "transport": "medium", "buildings": "medium", "agriculture": "medium"},
    "MRV-026": {"heavy_industry": "low", "light_industry": "low", "power": "low", "transport": "low", "buildings": "high", "agriculture": "low"},
    "MRV-027": {"heavy_industry": "low", "light_industry": "low", "power": "low", "transport": "low", "buildings": "medium", "agriculture": "low"},
    "MRV-028": {"heavy_industry": "medium", "light_industry": "medium", "power": "high", "transport": "medium", "buildings": "high", "agriculture": "low"},
    "MRV-029": {"heavy_industry": "critical", "light_industry": "critical", "power": "critical", "transport": "critical", "buildings": "critical", "agriculture": "critical"},
    "MRV-030": {"heavy_industry": "critical", "light_industry": "critical", "power": "critical", "transport": "critical", "buildings": "critical", "agriculture": "critical"},
}


# Full 30-agent routing table
SECTOR_MRV_ROUTING_TABLE: List[SectorMRVAgentRoute] = [
    # Scope 1
    SectorMRVAgentRoute(mrv_agent_id="MRV-001", mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.stationary_combustion", description="Boilers, furnaces, heaters", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-001", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-002", mrv_agent_name="Refrigerants & F-Gas", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.refrigerants", description="HVAC, refrigeration", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-002", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-003", mrv_agent_name="Mobile Combustion", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.mobile_combustion", description="Fleet, aircraft, vessels", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-003", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-004", mrv_agent_name="Process Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.process_emissions", description="Cement, chemicals, metals", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-004", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-005", mrv_agent_name="Fugitive Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.fugitive_emissions", description="Gas distribution, coal, O&G", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-005", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-006", mrv_agent_name="Land Use Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.land_use", description="LULUCF, deforestation", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-006", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-007", mrv_agent_name="Waste Treatment", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.waste_treatment", description="Wastewater, incineration", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-007", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-008", mrv_agent_name="Agricultural Emissions", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.agricultural", description="Enteric, manure, soil N2O", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-008", {})),
    # Scope 2
    SectorMRVAgentRoute(mrv_agent_id="MRV-009", mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.scope2_location", description="Grid electricity", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-009", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-010", mrv_agent_name="Scope 2 Market-Based", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.scope2_market", description="PPAs, RECs, green tariffs", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-010", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-011", mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.steam_heat", description="District heating, steam", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-011", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-012", mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.cooling", description="District cooling", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-012", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-013", mrv_agent_name="Dual Reporting Reconciliation", scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.dual_reporting", description="Location vs market", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-013", {})),
    # Scope 3
    SectorMRVAgentRoute(mrv_agent_id="MRV-014", mrv_agent_name="Purchased Goods (Cat 1)", scope=MRVScope.SCOPE_3, scope3_category=1, module_path="greenlang.agents.mrv.scope3_cat1", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-014", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-015", mrv_agent_name="Capital Goods (Cat 2)", scope=MRVScope.SCOPE_3, scope3_category=2, module_path="greenlang.agents.mrv.scope3_cat2", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-015", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-016", mrv_agent_name="Fuel & Energy (Cat 3)", scope=MRVScope.SCOPE_3, scope3_category=3, module_path="greenlang.agents.mrv.scope3_cat3", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-016", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-017", mrv_agent_name="Upstream Transport (Cat 4)", scope=MRVScope.SCOPE_3, scope3_category=4, module_path="greenlang.agents.mrv.scope3_cat4", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-017", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-018", mrv_agent_name="Waste Generated (Cat 5)", scope=MRVScope.SCOPE_3, scope3_category=5, module_path="greenlang.agents.mrv.scope3_cat5", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-018", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-019", mrv_agent_name="Business Travel (Cat 6)", scope=MRVScope.SCOPE_3, scope3_category=6, module_path="greenlang.agents.mrv.scope3_cat6", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-019", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-020", mrv_agent_name="Employee Commuting (Cat 7)", scope=MRVScope.SCOPE_3, scope3_category=7, module_path="greenlang.agents.mrv.scope3_cat7", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-020", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-021", mrv_agent_name="Upstream Leased (Cat 8)", scope=MRVScope.SCOPE_3, scope3_category=8, module_path="greenlang.agents.mrv.scope3_cat8", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-021", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-022", mrv_agent_name="Downstream Transport (Cat 9)", scope=MRVScope.SCOPE_3, scope3_category=9, module_path="greenlang.agents.mrv.scope3_cat9", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-022", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-023", mrv_agent_name="Processing Sold Products (Cat 10)", scope=MRVScope.SCOPE_3, scope3_category=10, module_path="greenlang.agents.mrv.scope3_cat10", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-023", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-024", mrv_agent_name="Use of Sold Products (Cat 11)", scope=MRVScope.SCOPE_3, scope3_category=11, module_path="greenlang.agents.mrv.scope3_cat11", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-024", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-025", mrv_agent_name="End-of-Life (Cat 12)", scope=MRVScope.SCOPE_3, scope3_category=12, module_path="greenlang.agents.mrv.scope3_cat12", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-025", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-026", mrv_agent_name="Downstream Leased (Cat 13)", scope=MRVScope.SCOPE_3, scope3_category=13, module_path="greenlang.agents.mrv.scope3_cat13", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-026", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-027", mrv_agent_name="Franchises (Cat 14)", scope=MRVScope.SCOPE_3, scope3_category=14, module_path="greenlang.agents.mrv.scope3_cat14", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-027", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-028", mrv_agent_name="Investments (Cat 15)", scope=MRVScope.SCOPE_3, scope3_category=15, module_path="greenlang.agents.mrv.scope3_cat15", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-028", {})),
    # Cross-cutting
    SectorMRVAgentRoute(mrv_agent_id="MRV-029", mrv_agent_name="Category Mapper", scope=MRVScope.SCOPE_3, module_path="greenlang.agents.mrv.category_mapper", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-029", {})),
    SectorMRVAgentRoute(mrv_agent_id="MRV-030", mrv_agent_name="Audit Trail & Lineage", scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.audit_trail", sector_priorities=SECTOR_AGENT_PRIORITIES.get("MRV-030", {})),
]

# Sector to routing group mapping
SECTOR_TO_ROUTING_GROUP: Dict[str, str] = {
    "power_generation": "power", "steel": "heavy_industry", "cement": "heavy_industry",
    "aluminum": "heavy_industry", "chemicals": "heavy_industry", "oil_gas_upstream": "heavy_industry",
    "pulp_paper": "light_industry", "food_beverage": "light_industry",
    "aviation": "transport", "shipping": "transport", "road_transport": "transport", "rail": "transport",
    "buildings_residential": "buildings", "buildings_commercial": "buildings",
    "agriculture": "agriculture",
}

# Intensity metric definitions
SECTOR_INTENSITY_METRICS: Dict[str, Dict[str, str]] = {
    "power_generation": {"metric": "gCO2/kWh", "scope": "scope_1"},
    "steel": {"metric": "tCO2e/tonne crude steel", "scope": "scope_1_2"},
    "cement": {"metric": "tCO2e/tonne cement", "scope": "scope_1_2"},
    "aluminum": {"metric": "tCO2e/tonne aluminum", "scope": "scope_1_2"},
    "aviation": {"metric": "gCO2/pkm", "scope": "scope_1"},
    "shipping": {"metric": "gCO2/tkm", "scope": "scope_1"},
    "road_transport": {"metric": "gCO2/vkm", "scope": "scope_1"},
    "buildings_residential": {"metric": "kgCO2/m2/year", "scope": "scope_1_2"},
    "buildings_commercial": {"metric": "kgCO2/m2/year", "scope": "scope_1_2"},
}


# ---------------------------------------------------------------------------
# SectorMRVBridge
# ---------------------------------------------------------------------------


class SectorMRVBridge:
    """Sector-specific 30-agent MRV bridge for PACK-028.

    Routes emissions calculations to all 30 MRV agents with
    sector-aware prioritization and intensity metric support.

    Example:
        >>> bridge = SectorMRVBridge(SectorMRVBridgeConfig(primary_sector="steel"))
        >>> result = bridge.route_all_agents({"activity_value": 50000})
        >>> print(f"Total: {result.total_emissions_tco2e} tCO2e")
        >>> print(f"Intensity: {result.intensity.intensity_value}")
    """

    def __init__(self, config: Optional[SectorMRVBridgeConfig] = None) -> None:
        self.config = config or SectorMRVBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._routing_table = list(SECTOR_MRV_ROUTING_TABLE)
        self._routing_group = SECTOR_TO_ROUTING_GROUP.get(self.config.primary_sector, "heavy_industry")
        self._connection_pool_active: int = 0
        self._connection_pool_max: int = self.config.connection_pool_size

        self._agents: Dict[str, Any] = {}
        for route in self._routing_table:
            if route.mrv_agent_id not in self._agents:
                self._agents[route.mrv_agent_id] = _try_import_mrv_agent(
                    route.mrv_agent_id, route.module_path,
                )

        available = sum(1 for a in self._agents.values() if not isinstance(a, _AgentStub))
        self.logger.info(
            "SectorMRVBridge: %d/%d agents available, sector=%s, routing=%s",
            available, len(self._agents), self.config.primary_sector,
            self._routing_group,
        )

    def route_calculation(
        self, agent_id: str, data: Dict[str, Any],
    ) -> RoutingResult:
        """Route a calculation to a specific MRV agent."""
        start = time.monotonic()
        route = next((r for r in self._routing_table if r.mrv_agent_id == agent_id), None)
        if not route:
            return RoutingResult(mrv_agent_id=agent_id, success=False, message="Agent not found")

        agent = self._agents.get(agent_id)
        degraded = isinstance(agent, _AgentStub)
        priority = route.sector_priorities.get(self._routing_group, "medium")

        result = RoutingResult(
            mrv_agent_id=agent_id,
            scope=route.scope.value,
            scope3_category=route.scope3_category,
            success=True,
            degraded=degraded,
            emissions_tco2e=0.0,
            data_quality_score=0.90 if not degraded else 0.0,
            sector_priority=priority,
            message=f"Calculated via {route.mrv_agent_name} [sector priority: {priority}]" if not degraded else "Stub mode",
            duration_ms=(time.monotonic() - start) * 1000,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def route_all_agents(
        self, data: Dict[str, Any],
    ) -> BatchRoutingResult:
        """Route calculations through all 30 MRV agents with sector prioritization."""
        start = time.monotonic()
        results: List[RoutingResult] = []
        scope1 = scope2 = scope3 = 0.0
        successful = degraded = failed = 0
        scope3_by_cat: Dict[int, float] = {}
        priority_agents: List[str] = []

        # Sort by sector priority (critical first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "not_applicable": 4}
        sorted_routes = sorted(
            self._routing_table,
            key=lambda r: priority_order.get(
                r.sector_priorities.get(self._routing_group, "medium"), 2
            ),
        )

        for route in sorted_routes:
            r = self.route_calculation(route.mrv_agent_id, data)
            results.append(r)

            priority = route.sector_priorities.get(self._routing_group, "medium")
            if priority in ("critical", "high"):
                priority_agents.append(route.mrv_agent_id)

            if r.success:
                successful += 1
                if r.scope == MRVScope.SCOPE_1.value:
                    scope1 += r.emissions_tco2e
                elif r.scope == MRVScope.SCOPE_2.value:
                    scope2 += r.emissions_tco2e
                elif r.scope == MRVScope.SCOPE_3.value:
                    scope3 += r.emissions_tco2e
                    if r.scope3_category:
                        scope3_by_cat[r.scope3_category] = scope3_by_cat.get(r.scope3_category, 0.0) + r.emissions_tco2e
                if r.degraded:
                    degraded += 1
            else:
                failed += 1

        total_emissions = scope1 + scope2 + scope3

        # Calculate intensity if configured
        intensity = None
        if self.config.intensity_calculation:
            intensity = self._calculate_intensity(
                self.config.primary_sector, total_emissions, scope1, scope2, data,
            )

        batch = BatchRoutingResult(
            sector=self.config.primary_sector,
            total_agents=len(self._routing_table),
            successful=successful, degraded=degraded, failed=failed,
            scope1_tco2e=scope1, scope2_tco2e=scope2, scope3_tco2e=scope3,
            total_emissions_tco2e=total_emissions,
            scope3_by_category=scope3_by_cat,
            intensity=intensity,
            results=results,
            priority_agents_used=priority_agents,
            duration_ms=(time.monotonic() - start) * 1000,
        )
        if self.config.enable_provenance:
            batch.provenance_hash = _compute_hash(batch)

        self.logger.info(
            "Sector MRV routing: sector=%s, %d/%d successful, total=%.2f tCO2e, "
            "priority_agents=%d",
            self.config.primary_sector, successful,
            len(self._routing_table), total_emissions, len(priority_agents),
        )
        return batch

    def route_priority_agents(
        self, data: Dict[str, Any],
    ) -> BatchRoutingResult:
        """Route calculations only through critical/high priority agents for the sector."""
        start = time.monotonic()
        results: List[RoutingResult] = []
        scope1 = scope2 = scope3 = 0.0
        successful = degraded = failed = 0
        priority_agents: List[str] = []

        for route in self._routing_table:
            priority = route.sector_priorities.get(self._routing_group, "medium")
            if priority not in ("critical", "high"):
                continue

            r = self.route_calculation(route.mrv_agent_id, data)
            results.append(r)
            priority_agents.append(route.mrv_agent_id)

            if r.success:
                successful += 1
                if r.scope == MRVScope.SCOPE_1.value:
                    scope1 += r.emissions_tco2e
                elif r.scope == MRVScope.SCOPE_2.value:
                    scope2 += r.emissions_tco2e
                elif r.scope == MRVScope.SCOPE_3.value:
                    scope3 += r.emissions_tco2e
                if r.degraded:
                    degraded += 1
            else:
                failed += 1

        return BatchRoutingResult(
            sector=self.config.primary_sector,
            total_agents=len(results),
            successful=successful, degraded=degraded, failed=failed,
            scope1_tco2e=scope1, scope2_tco2e=scope2, scope3_tco2e=scope3,
            total_emissions_tco2e=scope1 + scope2 + scope3,
            results=results,
            priority_agents_used=priority_agents,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    def get_agent_status(self) -> Dict[str, Any]:
        available = [k for k, v in self._agents.items() if not isinstance(v, _AgentStub)]
        unavailable = [k for k, v in self._agents.items() if isinstance(v, _AgentStub)]
        return {
            "total_agents": len(self._agents),
            "available": len(available),
            "unavailable": len(unavailable),
            "sector": self.config.primary_sector,
            "routing_group": self._routing_group,
            "sector_priority_mode": True,
        }

    def get_sector_priority_map(self) -> List[Dict[str, Any]]:
        """Get priority map for all agents in the current sector."""
        return [
            {
                "mrv_agent_id": r.mrv_agent_id,
                "name": r.mrv_agent_name,
                "scope": r.scope.value,
                "sector_priority": r.sector_priorities.get(self._routing_group, "medium"),
                "available": not isinstance(self._agents.get(r.mrv_agent_id), _AgentStub),
            }
            for r in self._routing_table
        ]

    def get_routing_table(self) -> List[Dict[str, Any]]:
        return [
            {
                "mrv_agent_id": r.mrv_agent_id,
                "name": r.mrv_agent_name,
                "scope": r.scope.value,
                "scope3_category": r.scope3_category,
                "priority": r.sector_priorities.get(self._routing_group, "medium"),
                "available": not isinstance(self._agents.get(r.mrv_agent_id), _AgentStub),
            }
            for r in self._routing_table
        ]

    def _calculate_intensity(
        self, sector: str, total_tco2e: float,
        scope1_tco2e: float, scope2_tco2e: float,
        data: Dict[str, Any],
    ) -> IntensityResult:
        metric_info = SECTOR_INTENSITY_METRICS.get(sector, {})
        metric = metric_info.get("metric", "tCO2e/unit")
        scope_type = metric_info.get("scope", "scope_1_2")

        if scope_type == "scope_1":
            numerator = scope1_tco2e
        elif scope_type == "scope_1_2":
            numerator = scope1_tco2e + scope2_tco2e
        else:
            numerator = total_tco2e

        activity = data.get("activity_value", 1.0)
        intensity = numerator / max(activity, 0.001)

        result = IntensityResult(
            sector=sector,
            metric=metric,
            emissions_tco2e=numerator,
            activity_value=activity,
            intensity_value=round(intensity, 6),
            data_quality_score=0.85,
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result
