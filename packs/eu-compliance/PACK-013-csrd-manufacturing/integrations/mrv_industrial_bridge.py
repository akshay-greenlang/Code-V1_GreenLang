"""
PACK-013 CSRD Manufacturing Pack - MRV Industrial Bridge.

Routes manufacturing emission-calculation requests to the appropriate
AGENT-MRV agents (001-030).  Includes a routing table that maps ESRS E1
disclosure codes and manufacturing-specific metric codes to MRV agent
IDs, along with convenience methods for Scope 1/2/3 calculations.
"""

import hashlib
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MRVBridgeConfig(BaseModel):
    """Configuration for the MRV industrial bridge."""
    enabled_agents: List[str] = Field(
        default_factory=lambda: [
            "MRV-001", "MRV-002", "MRV-003", "MRV-004", "MRV-005",
            "MRV-006", "MRV-007", "MRV-008", "MRV-009", "MRV-010",
            "MRV-011", "MRV-012", "MRV-013", "MRV-014", "MRV-015",
            "MRV-016", "MRV-017", "MRV-018", "MRV-019", "MRV-020",
            "MRV-021", "MRV-022", "MRV-023", "MRV-024", "MRV-025",
            "MRV-026", "MRV-027", "MRV-028", "MRV-029", "MRV-030",
        ]
    )
    routing_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Override default ESRS-to-agent routing",
    )
    timeout_ms: int = Field(default=60_000, ge=1_000)
    agent_module_prefix: str = Field(
        default="greenlang.agents.mrv"
    )
    fallback_to_stubs: bool = Field(default=True)


class MRVRouting(BaseModel):
    """Single routing entry mapping an ESRS code to an MRV agent."""
    esrs_code: str
    agent_id: str
    agent_module: str
    method_name: str
    description: str = Field(default="")


# ---------------------------------------------------------------------------
# Internal stub
# ---------------------------------------------------------------------------

class _MRVAgentStub:
    """Stub returned when an MRV agent cannot be imported."""

    def __init__(self, agent_id: str, reason: str = "not installed") -> None:
        self._agent_id = agent_id
        self._reason = reason

    def __getattr__(self, name: str) -> Callable[..., Dict[str, Any]]:
        def _stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "unavailable",
                "agent_id": self._agent_id,
                "method": name,
                "reason": self._reason,
                "total_tco2e": 0.0,
            }
        return _stub

    def __repr__(self) -> str:
        return f"_MRVAgentStub({self._agent_id!r})"


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

# Maps ESRS metric codes (and manufacturing-specific codes) to MRV agent
# IDs.  The orchestrator uses this to dispatch calculation requests.
DEFAULT_ROUTING_TABLE: List[Dict[str, str]] = [
    # -- Scope 1 ---
    {
        "esrs_code": "E1-6_scope1_stationary",
        "agent_id": "MRV-001",
        "agent_module": "agent_001_stationary_combustion",
        "method_name": "calculate",
        "description": "Stationary combustion (boilers, furnaces, kilns)",
    },
    {
        "esrs_code": "E1-6_scope1_refrigerants",
        "agent_id": "MRV-002",
        "agent_module": "agent_002_refrigerants_fgas",
        "method_name": "calculate",
        "description": "Refrigerants and F-gas losses",
    },
    {
        "esrs_code": "E1-6_scope1_mobile",
        "agent_id": "MRV-003",
        "agent_module": "agent_003_mobile_combustion",
        "method_name": "calculate",
        "description": "Mobile combustion (forklifts, vehicles)",
    },
    {
        "esrs_code": "E1-6_scope1_process",
        "agent_id": "MRV-004",
        "agent_module": "agent_004_process_emissions",
        "method_name": "calculate",
        "description": "Industrial process emissions (clinker, metal)",
    },
    {
        "esrs_code": "E1-6_scope1_fugitive",
        "agent_id": "MRV-005",
        "agent_module": "agent_005_fugitive_emissions",
        "method_name": "calculate",
        "description": "Fugitive emissions from equipment leaks",
    },
    {
        "esrs_code": "E1-6_scope1_landuse",
        "agent_id": "MRV-006",
        "agent_module": "agent_006_land_use",
        "method_name": "calculate",
        "description": "Land use change emissions",
    },
    {
        "esrs_code": "E1-6_scope1_waste_treatment",
        "agent_id": "MRV-007",
        "agent_module": "agent_007_waste_treatment",
        "method_name": "calculate",
        "description": "On-site waste treatment emissions",
    },
    {
        "esrs_code": "E1-6_scope1_agricultural",
        "agent_id": "MRV-008",
        "agent_module": "agent_008_agricultural",
        "method_name": "calculate",
        "description": "Agricultural emissions (food/bev manufacturing)",
    },
    # -- Scope 2 ---
    {
        "esrs_code": "E1-6_scope2_location",
        "agent_id": "MRV-009",
        "agent_module": "agent_009_scope2_location",
        "method_name": "calculate",
        "description": "Scope 2 location-based electricity emissions",
    },
    {
        "esrs_code": "E1-6_scope2_market",
        "agent_id": "MRV-010",
        "agent_module": "agent_010_scope2_market",
        "method_name": "calculate",
        "description": "Scope 2 market-based electricity emissions",
    },
    {
        "esrs_code": "E1-6_scope2_steam",
        "agent_id": "MRV-011",
        "agent_module": "agent_011_steam_heat",
        "method_name": "calculate",
        "description": "Purchased steam/heat emissions",
    },
    {
        "esrs_code": "E1-6_scope2_cooling",
        "agent_id": "MRV-012",
        "agent_module": "agent_012_cooling",
        "method_name": "calculate",
        "description": "Purchased cooling emissions",
    },
    {
        "esrs_code": "E1-6_scope2_dual",
        "agent_id": "MRV-013",
        "agent_module": "agent_013_dual_reporting",
        "method_name": "reconcile",
        "description": "Dual-reporting reconciliation",
    },
    # -- Scope 3 ---
    {
        "esrs_code": "E1-6_scope3_cat1",
        "agent_id": "MRV-014",
        "agent_module": "agent_014_purchased_goods",
        "method_name": "calculate",
        "description": "Cat 1 - Purchased goods and services",
    },
    {
        "esrs_code": "E1-6_scope3_cat2",
        "agent_id": "MRV-015",
        "agent_module": "agent_015_capital_goods",
        "method_name": "calculate",
        "description": "Cat 2 - Capital goods",
    },
    {
        "esrs_code": "E1-6_scope3_cat3",
        "agent_id": "MRV-016",
        "agent_module": "agent_016_fuel_energy",
        "method_name": "calculate",
        "description": "Cat 3 - Fuel and energy activities",
    },
    {
        "esrs_code": "E1-6_scope3_cat4",
        "agent_id": "MRV-017",
        "agent_module": "agent_017_upstream_transport",
        "method_name": "calculate",
        "description": "Cat 4 - Upstream transportation",
    },
    {
        "esrs_code": "E1-6_scope3_cat5",
        "agent_id": "MRV-018",
        "agent_module": "agent_018_waste_generated",
        "method_name": "calculate",
        "description": "Cat 5 - Waste generated in operations",
    },
    {
        "esrs_code": "E1-6_scope3_cat6",
        "agent_id": "MRV-019",
        "agent_module": "agent_019_business_travel",
        "method_name": "calculate",
        "description": "Cat 6 - Business travel",
    },
    {
        "esrs_code": "E1-6_scope3_cat7",
        "agent_id": "MRV-020",
        "agent_module": "agent_020_employee_commuting",
        "method_name": "calculate",
        "description": "Cat 7 - Employee commuting",
    },
    {
        "esrs_code": "E1-6_scope3_cat8",
        "agent_id": "MRV-021",
        "agent_module": "agent_021_upstream_leased",
        "method_name": "calculate",
        "description": "Cat 8 - Upstream leased assets",
    },
    {
        "esrs_code": "E1-6_scope3_cat9",
        "agent_id": "MRV-022",
        "agent_module": "agent_022_downstream_transport",
        "method_name": "calculate",
        "description": "Cat 9 - Downstream transportation",
    },
    {
        "esrs_code": "E1-6_scope3_cat10",
        "agent_id": "MRV-023",
        "agent_module": "agent_023_processing_sold",
        "method_name": "calculate",
        "description": "Cat 10 - Processing of sold products",
    },
    {
        "esrs_code": "E1-6_scope3_cat11",
        "agent_id": "MRV-024",
        "agent_module": "agent_024_use_sold",
        "method_name": "calculate",
        "description": "Cat 11 - Use of sold products",
    },
    {
        "esrs_code": "E1-6_scope3_cat12",
        "agent_id": "MRV-025",
        "agent_module": "agent_025_end_of_life",
        "method_name": "calculate",
        "description": "Cat 12 - End-of-life treatment",
    },
    {
        "esrs_code": "E1-6_scope3_cat13",
        "agent_id": "MRV-026",
        "agent_module": "agent_026_downstream_leased",
        "method_name": "calculate",
        "description": "Cat 13 - Downstream leased assets",
    },
    {
        "esrs_code": "E1-6_scope3_cat14",
        "agent_id": "MRV-027",
        "agent_module": "agent_027_franchises",
        "method_name": "calculate",
        "description": "Cat 14 - Franchises",
    },
    {
        "esrs_code": "E1-6_scope3_cat15",
        "agent_id": "MRV-028",
        "agent_module": "agent_028_investments",
        "method_name": "calculate",
        "description": "Cat 15 - Investments",
    },
    {
        "esrs_code": "E1-6_scope3_mapper",
        "agent_id": "MRV-029",
        "agent_module": "agent_029_scope3_mapper",
        "method_name": "map_categories",
        "description": "Scope 3 category mapper (cross-cutting)",
    },
    {
        "esrs_code": "E1-6_audit_trail",
        "agent_id": "MRV-030",
        "agent_module": "agent_030_audit_trail",
        "method_name": "record",
        "description": "Audit trail and lineage (cross-cutting)",
    },
]

# Sub-sector specific agent prioritisation
SUB_SECTOR_AGENTS: Dict[str, List[str]] = {
    "cement": ["MRV-001", "MRV-004", "MRV-005", "MRV-009"],
    "steel": ["MRV-001", "MRV-004", "MRV-005", "MRV-009", "MRV-011"],
    "chemicals": ["MRV-001", "MRV-004", "MRV-005", "MRV-002", "MRV-009"],
    "automotive": ["MRV-001", "MRV-003", "MRV-009", "MRV-014", "MRV-017"],
    "food_beverage": ["MRV-001", "MRV-008", "MRV-009", "MRV-007"],
    "electronics": ["MRV-001", "MRV-002", "MRV-009", "MRV-014"],
    "paper_pulp": ["MRV-001", "MRV-004", "MRV-009", "MRV-007"],
    "textiles": ["MRV-001", "MRV-009", "MRV-014", "MRV-007"],
    "general": ["MRV-001", "MRV-004", "MRV-009", "MRV-014"],
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class MRVIndustrialBridge:
    """
    Route emission-calculation requests to the correct AGENT-MRV agents.

    Provides a single entry point (``route_calculation``) that inspects
    the ESRS code and dispatches to the appropriate MRV agent, plus
    convenience helpers for Scope 1, 2, and 3 batch calculations.
    """

    def __init__(
        self, config: Optional[MRVBridgeConfig] = None
    ) -> None:
        self.config = config or MRVBridgeConfig()
        self._agents: Dict[str, Any] = {}
        self._routing: Dict[str, MRVRouting] = {}
        self._build_routing_table()
        self._load_agents()

    # -- setup ---------------------------------------------------------------

    def _build_routing_table(self) -> None:
        """Build the ESRS-to-agent routing index."""
        for entry in DEFAULT_ROUTING_TABLE:
            esrs_code = entry["esrs_code"]
            # Apply overrides
            agent_id = self.config.routing_overrides.get(
                esrs_code, entry["agent_id"]
            )
            self._routing[esrs_code] = MRVRouting(
                esrs_code=esrs_code,
                agent_id=agent_id,
                agent_module=entry["agent_module"],
                method_name=entry["method_name"],
                description=entry.get("description", ""),
            )

    def _load_agents(self) -> None:
        """Import all enabled MRV agents or fall back to stubs."""
        seen_modules: Dict[str, Any] = {}
        for route in self._routing.values():
            if route.agent_id not in self.config.enabled_agents:
                continue
            if route.agent_module in seen_modules:
                self._agents[route.agent_id] = seen_modules[
                    route.agent_module
                ]
                continue
            full_module = (
                f"{self.config.agent_module_prefix}."
                f"{route.agent_module}"
            )
            try:
                mod = importlib.import_module(full_module)
                self._agents[route.agent_id] = mod
                seen_modules[route.agent_module] = mod
                logger.info("Loaded MRV agent: %s", route.agent_id)
            except ImportError as exc:
                if self.config.fallback_to_stubs:
                    stub = _MRVAgentStub(route.agent_id, str(exc))
                    self._agents[route.agent_id] = stub
                    seen_modules[route.agent_module] = stub
                else:
                    logger.error(
                        "Cannot load MRV agent %s: %s",
                        route.agent_id, exc,
                    )

    def _get_agent(self, agent_id: str) -> Any:
        return self._agents.get(
            agent_id, _MRVAgentStub(agent_id, "not loaded")
        )

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- public API ----------------------------------------------------------

    def route_calculation(
        self,
        esrs_code: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Route a calculation request to the correct MRV agent based on
        the ESRS metric code.

        Args:
            esrs_code: ESRS disclosure code (e.g. ``E1-6_scope1_process``).
            input_data: Data payload for the calculation.

        Returns:
            Agent calculation result dict.
        """
        route = self._routing.get(esrs_code)
        if route is None:
            logger.warning("No routing for ESRS code: %s", esrs_code)
            return {
                "status": "no_route",
                "esrs_code": esrs_code,
                "total_tco2e": 0.0,
            }

        agent = self._get_agent(route.agent_id)
        method = getattr(agent, route.method_name, None)
        if method is None:
            return {
                "status": "method_not_found",
                "agent_id": route.agent_id,
                "method": route.method_name,
                "total_tco2e": 0.0,
            }

        try:
            result = method(input_data)
            result["agent_id"] = route.agent_id
            result["esrs_code"] = esrs_code
            return result
        except Exception as exc:
            logger.error(
                "MRV agent %s.%s failed: %s",
                route.agent_id, route.method_name, exc,
            )
            return {
                "status": "error",
                "agent_id": route.agent_id,
                "error": str(exc),
                "total_tco2e": 0.0,
            }

    def get_industrial_agent(
        self, sub_sector: str
    ) -> Any:
        """
        Return the primary MRV agent for a manufacturing sub-sector.

        Falls back to MRV-004 (process emissions) as the generic
        industrial agent.
        """
        priority = SUB_SECTOR_AGENTS.get(
            sub_sector, SUB_SECTOR_AGENTS["general"]
        )
        for agent_id in priority:
            agent = self._get_agent(agent_id)
            if not isinstance(agent, _MRVAgentStub):
                return agent
        # Return first stub if nothing real is available
        return self._get_agent(priority[0])

    def calculate_scope1(
        self, facility_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate total Scope 1 emissions by routing to MRV-001 through
        MRV-008 as appropriate.
        """
        scope1_codes = [
            "E1-6_scope1_stationary",
            "E1-6_scope1_process",
            "E1-6_scope1_fugitive",
            "E1-6_scope1_mobile",
            "E1-6_scope1_refrigerants",
        ]

        results: Dict[str, Any] = {}
        total = 0.0
        for code in scope1_codes:
            result = self.route_calculation(code, facility_data)
            tco2e = result.get("total_tco2e", 0.0)
            results[code] = result
            total += tco2e

        return {
            "scope1_total_tco2e": round(total, 4),
            "by_source": results,
            "provenance_hash": self._compute_hash(results),
        }

    def calculate_scope2(
        self, energy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate total Scope 2 emissions by routing to MRV-009 through
        MRV-013.
        """
        scope2_codes = [
            "E1-6_scope2_location",
            "E1-6_scope2_market",
            "E1-6_scope2_steam",
            "E1-6_scope2_cooling",
        ]

        results: Dict[str, Any] = {}
        location_total = 0.0
        market_total = 0.0
        for code in scope2_codes:
            result = self.route_calculation(code, energy_data)
            tco2e = result.get("total_tco2e", 0.0)
            results[code] = result
            if "location" in code or "steam" in code or "cooling" in code:
                location_total += tco2e
            if "market" in code:
                market_total += tco2e

        return {
            "scope2_location_tco2e": round(location_total, 4),
            "scope2_market_tco2e": round(market_total, 4),
            "by_source": results,
            "provenance_hash": self._compute_hash(results),
        }

    def calculate_scope3(
        self, supply_chain_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate total Scope 3 emissions by routing to MRV-014 through
        MRV-030 as needed.
        """
        scope3_codes = [
            f"E1-6_scope3_cat{i}" for i in range(1, 16)
        ]

        results: Dict[str, Any] = {}
        total = 0.0
        for code in scope3_codes:
            result = self.route_calculation(code, supply_chain_data)
            tco2e = result.get("total_tco2e", 0.0)
            results[code] = result
            total += tco2e

        return {
            "scope3_total_tco2e": round(total, 4),
            "by_category": results,
            "categories_calculated": len(results),
            "provenance_hash": self._compute_hash(results),
        }

    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all registered MRV agents and their availability."""
        agents_info: List[Dict[str, Any]] = []
        for route in self._routing.values():
            agent = self._agents.get(route.agent_id)
            agents_info.append({
                "agent_id": route.agent_id,
                "esrs_code": route.esrs_code,
                "module": route.agent_module,
                "method": route.method_name,
                "description": route.description,
                "available": (
                    agent is not None
                    and not isinstance(agent, _MRVAgentStub)
                ),
            })
        return agents_info
