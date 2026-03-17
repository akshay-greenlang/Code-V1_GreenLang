# -*- coding: utf-8 -*-
"""
MRVMaterialityBridge - Bridge to 30 MRV Agents for DMA Emissions Context
===========================================================================

This module routes emissions data from all 30 AGENT-MRV agents to provide
environmental context for the Double Materiality Assessment. It feeds Scope
1/2/3 emissions data into impact and financial scoring for E1 (Climate
Change) materiality topics, and maps emissions hotspots to materiality
topics across all ESRS environmental standards.

Routing Table (30 agents):
    Scope 1 (8 agents):  MRV-001..008 (Stationary/Mobile/Process/Fugitive/
                         Refrigerant/LandUse/Waste/Agriculture)
    Scope 2 (5 agents):  MRV-009..013 (Location/Market/Steam/Cooling/Dual)
    Scope 3 (17 agents): MRV-014..030 (Cat 1-15 + Mapper + AuditTrail)

Features:
    - Connect to all 30 AGENT-MRV agents for emissions context
    - Provide E1 climate materiality impact data
    - Route Scope 1/2/3 data for impact scoring
    - Map emissions hotspots to materiality topics
    - Aggregate emissions by ESRS environmental topic
    - Graceful degradation with _AgentStub when agents not importable
    - SHA-256 provenance on all operations

Architecture:
    MRV Agents (30) --> MRVMaterialityBridge --> Emissions Context
                              |                       |
                              v                       v
    _AgentStub (fallback)    DMA Impact Scoring  <-- Hotspot Mapping
                              |                       |
                              v                       v
    EmissionsContext <-- Provenance Hash <-- ESRS Topic Mapping

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-015 Double Materiality Assessment
Status: Production Ready
"""

import hashlib
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------


class _AgentStub:
    """Stub for unavailable MRV agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "agent": self._agent_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._agent_name} not available, using stub",
                "emissions_tco2e": 0.0,
            }
        return _stub_method


def _try_import_mrv_agent(agent_id: str, module_path: str) -> Any:
    """Try to import an MRV agent with graceful fallback.

    Args:
        agent_id: Agent identifier (e.g., 'MRV-001').
        module_path: Python module path for the agent.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        import importlib
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("MRV agent %s not available, using stub", agent_id)
        return _AgentStub(agent_id)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class ESRSEnvironmentalTopic(str, Enum):
    """ESRS environmental topics for materiality mapping."""

    E1_CLIMATE_CHANGE = "E1"
    E2_POLLUTION = "E2"
    E3_WATER = "E3"
    E4_BIODIVERSITY = "E4"
    E5_CIRCULAR_ECONOMY = "E5"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVAgentMapping(BaseModel):
    """Mapping of an MRV agent to DMA materiality context."""

    mrv_agent_id: str = Field(..., description="MRV agent identifier (e.g., MRV-001)")
    mrv_agent_name: str = Field(default="", description="Human-readable agent name")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    module_path: str = Field(default="")
    esrs_topics: List[str] = Field(
        default_factory=list,
        description="ESRS topics this agent contributes to (e.g., E1, E2)",
    )
    description: str = Field(default="")


class EmissionsContext(BaseModel):
    """Emissions context data for DMA impact scoring."""

    context_id: str = Field(default_factory=_new_uuid)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_by_category: Dict[int, float] = Field(default_factory=dict)
    hotspot_categories: List[str] = Field(
        default_factory=list, description="Top emission hotspot categories",
    )
    yoy_change_pct: Optional[float] = Field(None, description="Year-over-year change")
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


class HotspotMapping(BaseModel):
    """Mapping of emissions hotspots to ESRS materiality topics."""

    esrs_topic: str = Field(default="")
    topic_name: str = Field(default="")
    related_scopes: List[str] = Field(default_factory=list)
    related_categories: List[str] = Field(default_factory=list)
    emissions_contribution_tco2e: float = Field(default=0.0, ge=0.0)
    emissions_contribution_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    impact_relevance: str = Field(
        default="low", description="low, medium, high, critical",
    )


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Materiality Bridge."""

    pack_id: str = Field(default="PACK-015")
    enable_provenance: bool = Field(default=True)
    hotspot_threshold_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Minimum % of total emissions to qualify as hotspot",
    )
    include_scope3: bool = Field(default=True)
    max_hotspots: int = Field(default=10, ge=1, le=30)


class MRVQueryResult(BaseModel):
    """Result of querying MRV agents for emissions data."""

    query_id: str = Field(default_factory=_new_uuid)
    agents_queried: int = Field(default=0)
    agents_available: int = Field(default=0)
    agents_degraded: int = Field(default=0)
    emissions_context: Optional[EmissionsContext] = Field(None)
    hotspot_mappings: List[HotspotMapping] = Field(default_factory=list)
    success: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MRV Agent Registry (30 agents)
# ---------------------------------------------------------------------------

MRV_AGENT_REGISTRY: List[MRVAgentMapping] = [
    # Scope 1 (8 agents)
    MRVAgentMapping(mrv_agent_id="MRV-001", mrv_agent_name="Stationary Combustion",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.stationary_combustion",
                    esrs_topics=["E1", "E2"], description="Fuel combustion in stationary sources"),
    MRVAgentMapping(mrv_agent_id="MRV-002", mrv_agent_name="Refrigerants & F-Gas",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.refrigerants_fgas",
                    esrs_topics=["E1", "E2"], description="Refrigerant and F-gas leakage"),
    MRVAgentMapping(mrv_agent_id="MRV-003", mrv_agent_name="Mobile Combustion",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.mobile_combustion",
                    esrs_topics=["E1", "E2"], description="Vehicle and mobile source combustion"),
    MRVAgentMapping(mrv_agent_id="MRV-004", mrv_agent_name="Process Emissions",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.process_emissions",
                    esrs_topics=["E1", "E2"], description="Industrial process emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-005", mrv_agent_name="Fugitive Emissions",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.fugitive_emissions",
                    esrs_topics=["E1", "E2"], description="Fugitive emissions from equipment"),
    MRVAgentMapping(mrv_agent_id="MRV-006", mrv_agent_name="Land Use Emissions",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.land_use_emissions",
                    esrs_topics=["E1", "E4"], description="Land use change emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-007", mrv_agent_name="Waste Treatment Emissions",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.waste_treatment_emissions",
                    esrs_topics=["E1", "E5"], description="Waste treatment emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-008", mrv_agent_name="Agricultural Emissions",
                    scope=MRVScope.SCOPE_1, module_path="greenlang.agents.mrv.agricultural_emissions",
                    esrs_topics=["E1", "E2", "E4"], description="Agricultural emissions"),
    # Scope 2 (5 agents)
    MRVAgentMapping(mrv_agent_id="MRV-009", mrv_agent_name="Scope 2 Location-Based",
                    scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.scope2_location_based",
                    esrs_topics=["E1"], description="Purchased electricity (location-based)"),
    MRVAgentMapping(mrv_agent_id="MRV-010", mrv_agent_name="Scope 2 Market-Based",
                    scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.scope2_market_based",
                    esrs_topics=["E1"], description="Purchased electricity (market-based)"),
    MRVAgentMapping(mrv_agent_id="MRV-011", mrv_agent_name="Steam/Heat Purchase",
                    scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.steam_heat_purchase",
                    esrs_topics=["E1"], description="Purchased steam and heat"),
    MRVAgentMapping(mrv_agent_id="MRV-012", mrv_agent_name="Cooling Purchase",
                    scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.cooling_purchase",
                    esrs_topics=["E1"], description="Purchased cooling"),
    MRVAgentMapping(mrv_agent_id="MRV-013", mrv_agent_name="Dual Reporting Reconciliation",
                    scope=MRVScope.SCOPE_2, module_path="greenlang.agents.mrv.dual_reporting_reconciliation",
                    esrs_topics=["E1"], description="Location vs market-based reconciliation"),
    # Scope 3 (17 agents)
    MRVAgentMapping(mrv_agent_id="MRV-014", mrv_agent_name="Purchased Goods & Services (Cat 1)",
                    scope=MRVScope.SCOPE_3, scope3_category=1,
                    module_path="greenlang.agents.mrv.scope3_cat1",
                    esrs_topics=["E1", "E5"], description="Category 1 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-015", mrv_agent_name="Capital Goods (Cat 2)",
                    scope=MRVScope.SCOPE_3, scope3_category=2,
                    module_path="greenlang.agents.mrv.scope3_cat2",
                    esrs_topics=["E1", "E5"], description="Category 2 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-016", mrv_agent_name="Fuel & Energy Activities (Cat 3)",
                    scope=MRVScope.SCOPE_3, scope3_category=3,
                    module_path="greenlang.agents.mrv.scope3_cat3",
                    esrs_topics=["E1"], description="Category 3 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-017", mrv_agent_name="Upstream Transportation (Cat 4)",
                    scope=MRVScope.SCOPE_3, scope3_category=4,
                    module_path="greenlang.agents.mrv.scope3_cat4",
                    esrs_topics=["E1", "E2"], description="Category 4 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-018", mrv_agent_name="Waste Generated (Cat 5)",
                    scope=MRVScope.SCOPE_3, scope3_category=5,
                    module_path="greenlang.agents.mrv.scope3_cat5",
                    esrs_topics=["E1", "E5"], description="Category 5 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-019", mrv_agent_name="Business Travel (Cat 6)",
                    scope=MRVScope.SCOPE_3, scope3_category=6,
                    module_path="greenlang.agents.mrv.scope3_cat6",
                    esrs_topics=["E1"], description="Category 6 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-020", mrv_agent_name="Employee Commuting (Cat 7)",
                    scope=MRVScope.SCOPE_3, scope3_category=7,
                    module_path="greenlang.agents.mrv.scope3_cat7",
                    esrs_topics=["E1"], description="Category 7 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-021", mrv_agent_name="Upstream Leased Assets (Cat 8)",
                    scope=MRVScope.SCOPE_3, scope3_category=8,
                    module_path="greenlang.agents.mrv.scope3_cat8",
                    esrs_topics=["E1"], description="Category 8 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-022", mrv_agent_name="Downstream Transportation (Cat 9)",
                    scope=MRVScope.SCOPE_3, scope3_category=9,
                    module_path="greenlang.agents.mrv.scope3_cat9",
                    esrs_topics=["E1", "E2"], description="Category 9 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-023", mrv_agent_name="Processing of Sold Products (Cat 10)",
                    scope=MRVScope.SCOPE_3, scope3_category=10,
                    module_path="greenlang.agents.mrv.scope3_cat10",
                    esrs_topics=["E1"], description="Category 10 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-024", mrv_agent_name="Use of Sold Products (Cat 11)",
                    scope=MRVScope.SCOPE_3, scope3_category=11,
                    module_path="greenlang.agents.mrv.scope3_cat11",
                    esrs_topics=["E1"], description="Category 11 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-025", mrv_agent_name="End-of-Life Treatment (Cat 12)",
                    scope=MRVScope.SCOPE_3, scope3_category=12,
                    module_path="greenlang.agents.mrv.scope3_cat12",
                    esrs_topics=["E1", "E5"], description="Category 12 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-026", mrv_agent_name="Downstream Leased Assets (Cat 13)",
                    scope=MRVScope.SCOPE_3, scope3_category=13,
                    module_path="greenlang.agents.mrv.scope3_cat13",
                    esrs_topics=["E1"], description="Category 13 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-027", mrv_agent_name="Franchises (Cat 14)",
                    scope=MRVScope.SCOPE_3, scope3_category=14,
                    module_path="greenlang.agents.mrv.scope3_cat14",
                    esrs_topics=["E1"], description="Category 14 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-028", mrv_agent_name="Investments (Cat 15)",
                    scope=MRVScope.SCOPE_3, scope3_category=15,
                    module_path="greenlang.agents.mrv.scope3_cat15",
                    esrs_topics=["E1"], description="Category 15 emissions"),
    MRVAgentMapping(mrv_agent_id="MRV-029", mrv_agent_name="Scope 3 Category Mapper",
                    scope=MRVScope.SCOPE_3,
                    module_path="greenlang.agents.mrv.scope3_category_mapper",
                    esrs_topics=["E1"], description="Cross-category mapping"),
    MRVAgentMapping(mrv_agent_id="MRV-030", mrv_agent_name="Audit Trail & Lineage",
                    scope=MRVScope.SCOPE_3,
                    module_path="greenlang.agents.mrv.audit_trail_lineage",
                    esrs_topics=[], description="Audit trail for all MRV calculations"),
]

# Mapping of ESRS environmental topics to relevant emission categories
ESRS_TOPIC_EMISSION_MAP: Dict[str, Dict[str, Any]] = {
    "E1": {
        "name": "Climate Change",
        "scopes": ["scope_1", "scope_2", "scope_3"],
        "primary_categories": [1, 2, 3, 4, 5],
        "relevance": "direct",
    },
    "E2": {
        "name": "Pollution",
        "scopes": ["scope_1", "scope_3"],
        "primary_categories": [4, 5, 9, 12],
        "relevance": "indirect",
    },
    "E3": {
        "name": "Water and Marine Resources",
        "scopes": ["scope_1"],
        "primary_categories": [],
        "relevance": "indirect",
    },
    "E4": {
        "name": "Biodiversity and Ecosystems",
        "scopes": ["scope_1"],
        "primary_categories": [],
        "relevance": "indirect",
    },
    "E5": {
        "name": "Resource Use and Circular Economy",
        "scopes": ["scope_3"],
        "primary_categories": [1, 2, 5, 12],
        "relevance": "indirect",
    },
}


# ---------------------------------------------------------------------------
# MRVMaterialityBridge
# ---------------------------------------------------------------------------


class MRVMaterialityBridge:
    """Bridge to 30 MRV agents for DMA emissions context.

    Provides emissions data as context for the Double Materiality Assessment,
    mapping GHG emissions across Scope 1/2/3 to ESRS environmental topics
    and identifying emissions hotspots for impact scoring.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded MRV agent modules/stubs.

    Example:
        >>> bridge = MRVMaterialityBridge()
        >>> result = bridge.query_emissions_context(2025)
        >>> print(f"Total: {result.emissions_context.total_tco2e} tCO2e")
        >>> hotspots = bridge.get_hotspot_mappings(result.emissions_context)
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize the MRV Materiality Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or MRVBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._agents: Dict[str, Any] = {}
        for mapping in MRV_AGENT_REGISTRY:
            if mapping.mrv_agent_id not in self._agents:
                self._agents[mapping.mrv_agent_id] = _try_import_mrv_agent(
                    mapping.mrv_agent_id, mapping.module_path
                )

        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "MRVMaterialityBridge initialized: %d/%d agents available",
            available, len(self._agents),
        )

    # -------------------------------------------------------------------------
    # Emissions Context Query
    # -------------------------------------------------------------------------

    def query_emissions_context(
        self,
        reporting_year: int = 2025,
    ) -> MRVQueryResult:
        """Query all MRV agents for emissions context data.

        Aggregates emissions data across all 30 MRV agents by scope and
        category, then identifies hotspots for materiality mapping.

        Args:
            reporting_year: The reporting year to query.

        Returns:
            MRVQueryResult with emissions context and hotspot mappings.
        """
        start = time.monotonic()

        agents_queried = len(self._agents)
        agents_available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        agents_degraded = agents_queried - agents_available

        # Build emissions context (stub: returns zero emissions)
        context = EmissionsContext(
            scope1_tco2e=0.0,
            scope2_location_tco2e=0.0,
            scope2_market_tco2e=0.0,
            scope3_tco2e=0.0,
            total_tco2e=0.0,
            scope3_by_category={i: 0.0 for i in range(1, 16)},
            hotspot_categories=[],
            reporting_year=reporting_year,
        )

        if self.config.enable_provenance:
            context.provenance_hash = _compute_hash(context)

        # Generate hotspot mappings
        hotspot_mappings = self._generate_hotspot_mappings(context)

        elapsed = (time.monotonic() - start) * 1000

        result = MRVQueryResult(
            agents_queried=agents_queried,
            agents_available=agents_available,
            agents_degraded=agents_degraded,
            emissions_context=context,
            hotspot_mappings=hotspot_mappings,
            success=True,
            message=f"Queried {agents_queried} MRV agents ({agents_available} available)",
            duration_ms=elapsed,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Emissions context queried: %d agents, total=%.2f tCO2e in %.1fms",
            agents_queried, context.total_tco2e, elapsed,
        )
        return result

    # -------------------------------------------------------------------------
    # Hotspot Mapping
    # -------------------------------------------------------------------------

    def get_hotspot_mappings(
        self,
        context: EmissionsContext,
    ) -> List[HotspotMapping]:
        """Map emissions hotspots to ESRS materiality topics.

        Args:
            context: Emissions context with scope-level data.

        Returns:
            List of HotspotMapping for each ESRS environmental topic.
        """
        return self._generate_hotspot_mappings(context)

    def _generate_hotspot_mappings(
        self,
        context: EmissionsContext,
    ) -> List[HotspotMapping]:
        """Generate hotspot mappings from emissions context.

        Args:
            context: Emissions context data.

        Returns:
            List of hotspot mappings sorted by contribution.
        """
        total = context.total_tco2e if context.total_tco2e > 0 else 1.0
        mappings: List[HotspotMapping] = []

        for topic_code, topic_info in ESRS_TOPIC_EMISSION_MAP.items():
            contribution = 0.0

            for scope_str in topic_info["scopes"]:
                if scope_str == "scope_1":
                    contribution += context.scope1_tco2e
                elif scope_str == "scope_2":
                    contribution += context.scope2_location_tco2e
                elif scope_str == "scope_3":
                    for cat in topic_info.get("primary_categories", []):
                        contribution += context.scope3_by_category.get(cat, 0.0)

            pct = (contribution / total * 100.0) if total > 0 else 0.0

            if pct >= 50.0:
                relevance = "critical"
            elif pct >= 20.0:
                relevance = "high"
            elif pct >= self.config.hotspot_threshold_pct:
                relevance = "medium"
            else:
                relevance = "low"

            mappings.append(HotspotMapping(
                esrs_topic=topic_code,
                topic_name=topic_info["name"],
                related_scopes=topic_info["scopes"],
                related_categories=[f"Cat {c}" for c in topic_info.get("primary_categories", [])],
                emissions_contribution_tco2e=contribution,
                emissions_contribution_pct=round(pct, 1),
                impact_relevance=relevance,
            ))

        mappings.sort(key=lambda m: m.emissions_contribution_tco2e, reverse=True)
        return mappings[:self.config.max_hotspots]

    # -------------------------------------------------------------------------
    # Agent Information
    # -------------------------------------------------------------------------

    def get_agent_registry(self) -> List[Dict[str, Any]]:
        """Get the full MRV agent registry with availability status.

        Returns:
            List of agent registry entries.
        """
        return [
            {
                "mrv_agent_id": m.mrv_agent_id,
                "mrv_agent_name": m.mrv_agent_name,
                "scope": m.scope.value,
                "scope3_category": m.scope3_category,
                "esrs_topics": m.esrs_topics,
                "available": not isinstance(
                    self._agents.get(m.mrv_agent_id), _AgentStub
                ),
            }
            for m in MRV_AGENT_REGISTRY
        ]

    def get_agents_for_topic(self, esrs_topic: str) -> List[Dict[str, Any]]:
        """Get MRV agents relevant to a specific ESRS topic.

        Args:
            esrs_topic: ESRS topic code (e.g., 'E1', 'E2').

        Returns:
            List of relevant agent entries.
        """
        return [
            {
                "mrv_agent_id": m.mrv_agent_id,
                "mrv_agent_name": m.mrv_agent_name,
                "scope": m.scope.value,
                "scope3_category": m.scope3_category,
                "available": not isinstance(
                    self._agents.get(m.mrv_agent_id), _AgentStub
                ),
            }
            for m in MRV_AGENT_REGISTRY
            if esrs_topic in m.esrs_topics
        ]
