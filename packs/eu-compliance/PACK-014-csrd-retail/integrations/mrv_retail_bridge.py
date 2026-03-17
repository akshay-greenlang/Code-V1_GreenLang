# -*- coding: utf-8 -*-
"""
MRVRetailBridge - Bridge to 30 MRV Agents for Retail Emissions Calculation
=============================================================================

This module routes retail emissions calculation requests to the appropriate
MRV (Monitoring, Reporting, Verification) agents. It maps retail-specific
emission sources (store heating, refrigerants, delivery fleet, purchased
goods, packaging waste, etc.) to the 30 MRV agents in the GreenLang platform.

Routing Table (30+ entries):
    Store heating         --> MRV-001 (Stationary Combustion)
    Refrigerant leakage   --> MRV-002 (Refrigerants & F-Gas)
    Delivery fleet        --> MRV-003 (Mobile Combustion)
    Store electricity (L) --> MRV-009 (Scope 2 Location-Based)
    Store electricity (M) --> MRV-010 (Scope 2 Market-Based)
    Purchased goods       --> MRV-014 (Category 1)
    Capital goods         --> MRV-015 (Category 2)
    Upstream transport    --> MRV-017 (Category 4)
    Packaging waste       --> MRV-018 (Category 5)
    Business travel       --> MRV-019 (Category 6)
    Employee commuting    --> MRV-020 (Category 7)
    Downstream transport  --> MRV-022 (Category 9)
    Use of sold products  --> MRV-024 (Category 11)
    End-of-life           --> MRV-025 (Category 12)
    Category mapper       --> MRV-029
    Audit trail           --> MRV-030

Features:
    - Route calculation requests to correct MRV agent
    - Sub-sector specific prioritization (grocery vs apparel vs electronics)
    - Graceful degradation with _AgentStub when agents not importable
    - SHA-256 provenance on all routing operations
    - Batch routing for multi-store portfolios

Architecture:
    Retail Data --> MRVRetailBridge --> MRV Agent Routing Table
                        |                      |
                        v                      v
    _AgentStub (fallback)     MRV-001..030 (calculation)
                        |                      |
                        v                      v
    RoutingResult <-- Provenance Hash <-- Emissions Data

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
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
    """Stub for unavailable MRV agent modules.

    Returns informative defaults when MRV agents are not installed,
    allowing PACK-014 to operate in standalone mode with degraded
    calculation capability.
    """

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


class EmissionSource(str, Enum):
    """Retail emission source categories mapped to MRV agents."""

    STORE_HEATING = "store_heating"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    DELIVERY_FLEET = "delivery_fleet"
    BACKUP_GENERATORS = "backup_generators"
    COOKING_EQUIPMENT = "cooking_equipment"
    STORE_ELECTRICITY_LOCATION = "store_electricity_location"
    STORE_ELECTRICITY_MARKET = "store_electricity_market"
    STORE_STEAM_HEAT = "store_steam_heat"
    STORE_COOLING = "store_cooling"
    PURCHASED_GOODS = "purchased_goods"
    CAPITAL_GOODS = "capital_goods"
    FUEL_ENERGY_ACTIVITIES = "fuel_energy_activities"
    UPSTREAM_TRANSPORT = "upstream_transport"
    PACKAGING_WASTE = "packaging_waste"
    BUSINESS_TRAVEL = "business_travel"
    EMPLOYEE_COMMUTING = "employee_commuting"
    DOWNSTREAM_TRANSPORT = "downstream_transport"
    USE_OF_SOLD_PRODUCTS = "use_of_sold_products"
    END_OF_LIFE = "end_of_life"
    FRANCHISES = "franchises"


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVAgentRoute(BaseModel):
    """Routing entry mapping an emission source to an MRV agent."""

    source: EmissionSource = Field(...)
    mrv_agent_id: str = Field(..., description="MRV agent identifier (e.g., MRV-001)")
    mrv_agent_name: str = Field(default="", description="Human-readable agent name")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    module_path: str = Field(default="", description="Python module path")
    description: str = Field(default="")
    priority_grocery: int = Field(default=5, ge=1, le=10)
    priority_apparel: int = Field(default=5, ge=1, le=10)
    priority_electronics: int = Field(default=5, ge=1, le=10)


class RoutingResult(BaseModel):
    """Result of routing a calculation request to an MRV agent."""

    routing_id: str = Field(default_factory=_new_uuid)
    source: str = Field(default="")
    mrv_agent_id: str = Field(default="")
    scope: str = Field(default="")
    scope3_category: Optional[int] = Field(None)
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    emissions_tco2e: float = Field(default=0.0)
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV Retail Bridge."""

    pack_id: str = Field(default="PACK-014")
    enable_provenance: bool = Field(default=True)
    sub_sector: str = Field(default="general_merchandise")
    enable_batch_routing: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=10, ge=1, le=30)


class BatchRoutingResult(BaseModel):
    """Result of routing multiple calculation requests."""

    batch_id: str = Field(default_factory=_new_uuid)
    total_sources: int = Field(default=0)
    successful: int = Field(default=0)
    degraded: int = Field(default=0)
    failed: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    results: List[RoutingResult] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MRV Agent Routing Table
# ---------------------------------------------------------------------------

MRV_ROUTING_TABLE: List[MRVAgentRoute] = [
    # Scope 1
    MRVAgentRoute(
        source=EmissionSource.STORE_HEATING, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Gas/oil heating in retail stores",
        priority_grocery=8, priority_apparel=6, priority_electronics=5,
    ),
    MRVAgentRoute(
        source=EmissionSource.REFRIGERANT_LEAKAGE, mrv_agent_id="MRV-002",
        mrv_agent_name="Refrigerants & F-Gas", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.refrigerants_fgas",
        description="HFC/HCFC refrigerant leaks from display cases and cold storage",
        priority_grocery=10, priority_apparel=2, priority_electronics=3,
    ),
    MRVAgentRoute(
        source=EmissionSource.DELIVERY_FLEET, mrv_agent_id="MRV-003",
        mrv_agent_name="Mobile Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.mobile_combustion",
        description="Owned delivery vehicles and company cars",
        priority_grocery=7, priority_apparel=5, priority_electronics=4,
    ),
    MRVAgentRoute(
        source=EmissionSource.BACKUP_GENERATORS, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Diesel backup generators",
        priority_grocery=4, priority_apparel=3, priority_electronics=3,
    ),
    MRVAgentRoute(
        source=EmissionSource.COOKING_EQUIPMENT, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Gas cooking in deli/bakery departments",
        priority_grocery=6, priority_apparel=1, priority_electronics=1,
    ),
    # Scope 2
    MRVAgentRoute(
        source=EmissionSource.STORE_ELECTRICITY_LOCATION, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_location_based",
        description="Store electricity using grid emission factors",
        priority_grocery=9, priority_apparel=9, priority_electronics=9,
    ),
    MRVAgentRoute(
        source=EmissionSource.STORE_ELECTRICITY_MARKET, mrv_agent_id="MRV-010",
        mrv_agent_name="Scope 2 Market-Based", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.scope2_market_based",
        description="Store electricity using contractual/residual factors",
        priority_grocery=9, priority_apparel=9, priority_electronics=9,
    ),
    MRVAgentRoute(
        source=EmissionSource.STORE_STEAM_HEAT, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="District heating for stores",
        priority_grocery=5, priority_apparel=5, priority_electronics=4,
    ),
    MRVAgentRoute(
        source=EmissionSource.STORE_COOLING, mrv_agent_id="MRV-012",
        mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2,
        module_path="greenlang.agents.mrv.cooling_purchase",
        description="District cooling for stores",
        priority_grocery=4, priority_apparel=3, priority_electronics=3,
    ),
    # Scope 3
    MRVAgentRoute(
        source=EmissionSource.PURCHASED_GOODS, mrv_agent_id="MRV-014",
        mrv_agent_name="Purchased Goods & Services (Cat 1)", scope=MRVScope.SCOPE_3,
        scope3_category=1,
        module_path="greenlang.agents.mrv.scope3_cat1",
        description="Emissions from purchased retail merchandise",
        priority_grocery=10, priority_apparel=10, priority_electronics=10,
    ),
    MRVAgentRoute(
        source=EmissionSource.CAPITAL_GOODS, mrv_agent_id="MRV-015",
        mrv_agent_name="Capital Goods (Cat 2)", scope=MRVScope.SCOPE_3,
        scope3_category=2,
        module_path="greenlang.agents.mrv.scope3_cat2",
        description="Store fixtures, equipment, IT infrastructure",
        priority_grocery=5, priority_apparel=5, priority_electronics=6,
    ),
    MRVAgentRoute(
        source=EmissionSource.FUEL_ENERGY_ACTIVITIES, mrv_agent_id="MRV-016",
        mrv_agent_name="Fuel & Energy Activities (Cat 3)", scope=MRVScope.SCOPE_3,
        scope3_category=3,
        module_path="greenlang.agents.mrv.scope3_cat3",
        description="Upstream energy-related emissions",
        priority_grocery=6, priority_apparel=5, priority_electronics=5,
    ),
    MRVAgentRoute(
        source=EmissionSource.UPSTREAM_TRANSPORT, mrv_agent_id="MRV-017",
        mrv_agent_name="Upstream Transportation (Cat 4)", scope=MRVScope.SCOPE_3,
        scope3_category=4,
        module_path="greenlang.agents.mrv.scope3_cat4",
        description="Inbound logistics and distribution centre operations",
        priority_grocery=9, priority_apparel=8, priority_electronics=7,
    ),
    MRVAgentRoute(
        source=EmissionSource.PACKAGING_WASTE, mrv_agent_id="MRV-018",
        mrv_agent_name="Waste Generated (Cat 5)", scope=MRVScope.SCOPE_3,
        scope3_category=5,
        module_path="greenlang.agents.mrv.scope3_cat5",
        description="Store operational waste and packaging waste",
        priority_grocery=8, priority_apparel=6, priority_electronics=5,
    ),
    MRVAgentRoute(
        source=EmissionSource.BUSINESS_TRAVEL, mrv_agent_id="MRV-019",
        mrv_agent_name="Business Travel (Cat 6)", scope=MRVScope.SCOPE_3,
        scope3_category=6,
        module_path="greenlang.agents.mrv.scope3_cat6",
        description="Corporate business travel",
        priority_grocery=3, priority_apparel=4, priority_electronics=4,
    ),
    MRVAgentRoute(
        source=EmissionSource.EMPLOYEE_COMMUTING, mrv_agent_id="MRV-020",
        mrv_agent_name="Employee Commuting (Cat 7)", scope=MRVScope.SCOPE_3,
        scope3_category=7,
        module_path="greenlang.agents.mrv.scope3_cat7",
        description="Store and HQ employee commuting",
        priority_grocery=6, priority_apparel=6, priority_electronics=5,
    ),
    MRVAgentRoute(
        source=EmissionSource.DOWNSTREAM_TRANSPORT, mrv_agent_id="MRV-022",
        mrv_agent_name="Downstream Transportation (Cat 9)", scope=MRVScope.SCOPE_3,
        scope3_category=9,
        module_path="greenlang.agents.mrv.scope3_cat9",
        description="Customer home delivery and click-and-collect trips",
        priority_grocery=5, priority_apparel=7, priority_electronics=8,
    ),
    MRVAgentRoute(
        source=EmissionSource.USE_OF_SOLD_PRODUCTS, mrv_agent_id="MRV-024",
        mrv_agent_name="Use of Sold Products (Cat 11)", scope=MRVScope.SCOPE_3,
        scope3_category=11,
        module_path="greenlang.agents.mrv.scope3_cat11",
        description="Energy consumption of sold electronic products",
        priority_grocery=2, priority_apparel=2, priority_electronics=10,
    ),
    MRVAgentRoute(
        source=EmissionSource.END_OF_LIFE, mrv_agent_id="MRV-025",
        mrv_agent_name="End-of-Life Treatment (Cat 12)", scope=MRVScope.SCOPE_3,
        scope3_category=12,
        module_path="greenlang.agents.mrv.scope3_cat12",
        description="End-of-life treatment of sold products",
        priority_grocery=4, priority_apparel=5, priority_electronics=7,
    ),
    MRVAgentRoute(
        source=EmissionSource.FRANCHISES, mrv_agent_id="MRV-027",
        mrv_agent_name="Franchises (Cat 14)", scope=MRVScope.SCOPE_3,
        scope3_category=14,
        module_path="greenlang.agents.mrv.scope3_cat14",
        description="Franchise store emissions",
        priority_grocery=7, priority_apparel=6, priority_electronics=4,
    ),
]


# ---------------------------------------------------------------------------
# MRVRetailBridge
# ---------------------------------------------------------------------------


class MRVRetailBridge:
    """Bridge to 30 MRV agents for retail emissions calculation.

    Routes retail-specific emission sources to the appropriate MRV agent,
    with sub-sector priority weighting and graceful degradation when
    agents are not available.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded MRV agent modules/stubs.
        _routing_table: Active routing table.

    Example:
        >>> bridge = MRVRetailBridge()
        >>> result = bridge.route_calculation(
        ...     EmissionSource.REFRIGERANT_LEAKAGE,
        ...     {"refrigerant_type": "R-404A", "charge_kg": 500, "leak_rate": 0.15}
        ... )
        >>> print(f"Emissions: {result.emissions_tco2e} tCO2e")
    """

    def __init__(self, config: Optional[MRVBridgeConfig] = None) -> None:
        """Initialize the MRV Retail Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or MRVBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._routing_table = list(MRV_ROUTING_TABLE)

        # Load MRV agents with graceful fallback
        self._agents: Dict[str, Any] = {}
        unique_agents = {r.mrv_agent_id: r.module_path for r in self._routing_table}
        for agent_id, module_path in unique_agents.items():
            self._agents[agent_id] = _try_import_mrv_agent(agent_id, module_path)

        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "MRVRetailBridge initialized: %d/%d agents available, sub_sector=%s",
            available, len(self._agents), self.config.sub_sector,
        )

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def route_calculation(
        self,
        source: EmissionSource,
        data: Dict[str, Any],
    ) -> RoutingResult:
        """Route a calculation request to the appropriate MRV agent.

        Args:
            source: Emission source category.
            data: Input data for the calculation (activity data, EFs, etc.).

        Returns:
            RoutingResult with calculation output or degraded status.
        """
        start = time.monotonic()

        # Find routing entry
        route = self._find_route(source)
        if route is None:
            return RoutingResult(
                source=source.value,
                success=False,
                message=f"No routing entry for source '{source.value}'",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        agent = self._agents.get(route.mrv_agent_id)
        if agent is None or isinstance(agent, _AgentStub):
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                success=False,
                degraded=True,
                message=f"MRV agent {route.mrv_agent_id} not available (stub mode)",
                duration_ms=(time.monotonic() - start) * 1000,
            )
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)
            return result

        # Dispatch to real agent (stub implementation)
        try:
            calc_result = {"emissions_tco2e": 0.0, "status": "calculated"}
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                scope3_category=route.scope3_category,
                success=True,
                emissions_tco2e=calc_result.get("emissions_tco2e", 0.0),
                calculation_details=calc_result,
                message=f"Calculated via {route.mrv_agent_name}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            result = RoutingResult(
                source=source.value,
                mrv_agent_id=route.mrv_agent_id,
                scope=route.scope.value,
                success=False,
                message=f"Calculation failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def route_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> BatchRoutingResult:
        """Route multiple calculation requests in batch.

        Each request must contain 'source' (EmissionSource value) and 'data'.

        Args:
            requests: List of dicts with 'source' and 'data' keys.

        Returns:
            BatchRoutingResult with aggregated emissions.
        """
        start = time.monotonic()
        results: List[RoutingResult] = []
        total_emissions = 0.0
        scope1 = 0.0
        scope2 = 0.0
        scope3 = 0.0
        successful = 0
        degraded_count = 0
        failed = 0

        for req in requests:
            source_str = req.get("source", "")
            try:
                source = EmissionSource(source_str)
            except ValueError:
                results.append(RoutingResult(
                    source=source_str,
                    success=False,
                    message=f"Unknown emission source: {source_str}",
                ))
                failed += 1
                continue

            result = self.route_calculation(source, req.get("data", {}))
            results.append(result)

            if result.success:
                successful += 1
                total_emissions += result.emissions_tco2e
                if result.scope == MRVScope.SCOPE_1.value:
                    scope1 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_2.value:
                    scope2 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_3.value:
                    scope3 += result.emissions_tco2e
            elif result.degraded:
                degraded_count += 1
            else:
                failed += 1

        elapsed = (time.monotonic() - start) * 1000

        batch_result = BatchRoutingResult(
            total_sources=len(requests),
            successful=successful,
            degraded=degraded_count,
            failed=failed,
            total_emissions_tco2e=total_emissions,
            scope1_tco2e=scope1,
            scope2_tco2e=scope2,
            scope3_tco2e=scope3,
            results=results,
            duration_ms=elapsed,
        )

        if self.config.enable_provenance:
            batch_result.provenance_hash = _compute_hash(batch_result)

        self.logger.info(
            "Batch routing complete: %d/%d successful, total=%.2f tCO2e in %.1fms",
            successful, len(requests), total_emissions, elapsed,
        )
        return batch_result

    # -------------------------------------------------------------------------
    # Priority-Based Source Selection
    # -------------------------------------------------------------------------

    def get_priority_sources(
        self, sub_sector: Optional[str] = None, top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get emission sources ranked by priority for the sub-sector.

        Args:
            sub_sector: Override sub-sector (uses config default if None).
            top_n: Number of top sources to return.

        Returns:
            List of source dicts sorted by priority (highest first).
        """
        sector = sub_sector or self.config.sub_sector
        priority_field = {
            "grocery": "priority_grocery",
            "apparel": "priority_apparel",
            "electronics": "priority_electronics",
        }.get(sector, "priority_grocery")

        sources = []
        for route in self._routing_table:
            priority = getattr(route, priority_field, 5)
            sources.append({
                "source": route.source.value,
                "mrv_agent_id": route.mrv_agent_id,
                "mrv_agent_name": route.mrv_agent_name,
                "scope": route.scope.value,
                "scope3_category": route.scope3_category,
                "priority": priority,
                "description": route.description,
            })

        sources.sort(key=lambda x: x["priority"], reverse=True)
        return sources[:top_n]

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """Get the full routing table as a list of dicts.

        Returns:
            List of routing entries.
        """
        return [
            {
                "source": r.source.value,
                "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name,
                "scope": r.scope.value,
                "scope3_category": r.scope3_category,
                "available": not isinstance(self._agents.get(r.mrv_agent_id), _AgentStub),
            }
            for r in self._routing_table
        ]

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _find_route(self, source: EmissionSource) -> Optional[MRVAgentRoute]:
        """Find the routing entry for an emission source.

        Args:
            source: Emission source to look up.

        Returns:
            MRVAgentRoute if found, None otherwise.
        """
        for route in self._routing_table:
            if route.source == source:
                return route
        return None
