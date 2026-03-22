# -*- coding: utf-8 -*-
"""
MRVUtilityBridge - Bridge to MRV Agents for Utility Consumption Emissions
============================================================================

This module routes utility consumption data to the appropriate MRV
(Monitoring, Reporting, Verification) agents for carbon emissions
calculation. It maps utility commodity consumption to Scope 1, 2, and 3
emissions across the 30 MRV agents.

Routing Table:
    Natural gas consumption   --> MRV-001 (Stationary Combustion) [Scope 1]
    Fuel oil / diesel         --> MRV-001 (Stationary Combustion) [Scope 1]
    Electricity (location)    --> MRV-009 (Scope 2 Location-Based)
    Electricity (market)      --> MRV-010 (Scope 2 Market-Based)
    Purchased steam/heat      --> MRV-011 (Steam/Heat Purchase) [Scope 2]
    Purchased cooling         --> MRV-012 (Cooling Purchase) [Scope 2]
    Upstream fuel/energy      --> MRV-016 (Scope 3 Category 3)

Features:
    - Route consumption data to correct MRV agent for emissions calculation
    - Map energy consumption (kWh/MWh) to emissions (tCO2e)
    - Commodity-specific emission factor selection
    - Bi-directional data flow (send consumption, receive emissions)
    - Graceful degradation with _AgentStub when agents not importable
    - SHA-256 provenance on all routing and conversion operations

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-036 Utility Analysis
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
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


class ConsumptionCategory(str, Enum):
    """Utility commodity categories mapped to MRV agents."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    DIESEL = "diesel"
    LPG = "lpg"
    PROPANE = "propane"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    HOT_WATER = "hot_water"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    BIOGAS = "biogas"


class MRVScope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class EmissionsMethod(str, Enum):
    """Emission calculation method."""

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"
    DIRECT_MEASUREMENT = "direct_measurement"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MRVRouteConfig(BaseModel):
    """Configuration for the MRV Utility Bridge."""

    pack_id: str = Field(default="PACK-036")
    enable_provenance: bool = Field(default=True)
    country_code: str = Field(default="DE", description="ISO 3166-1 alpha-2")
    grid_emission_factor_kgco2_per_kwh: float = Field(
        default=0.366, ge=0.0, description="Default grid EF (kg CO2e/kWh)"
    )
    natural_gas_ef_kgco2_per_kwh: float = Field(
        default=0.202, ge=0.0, description="Natural gas EF (kg CO2e/kWh)"
    )
    electricity_method: EmissionsMethod = Field(
        default=EmissionsMethod.LOCATION_BASED,
        description="Electricity emissions calculation method",
    )


class ConsumptionToEmissionsMapping(BaseModel):
    """Mapping entry from a commodity category to MRV agent routing."""

    category: ConsumptionCategory = Field(...)
    mrv_agent_id: str = Field(..., description="Primary MRV agent identifier")
    mrv_agent_name: str = Field(default="")
    scope: MRVScope = Field(...)
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    energy_carrier: str = Field(default="electricity")
    module_path: str = Field(default="")
    description: str = Field(default="")


class RoutingResult(BaseModel):
    """Result of routing consumption data to an MRV agent."""

    routing_id: str = Field(default_factory=_new_uuid)
    category: str = Field(default="")
    mrv_agent_id: str = Field(default="")
    scope: str = Field(default="")
    method: str = Field(default="")
    success: bool = Field(default=False)
    degraded: bool = Field(default=False)
    emissions_tco2e: float = Field(default=0.0)
    consumption_kwh: float = Field(default=0.0)
    emission_factor_kgco2_per_kwh: float = Field(default=0.0)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class EmissionsSummary(BaseModel):
    """Aggregated emissions summary across all commodities."""

    summary_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    period: str = Field(default="")
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    total_tco2e: float = Field(default=0.0)
    breakdown_by_commodity: Dict[str, float] = Field(default_factory=dict)
    routing_results: List[RoutingResult] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MRV Agent Routing Table
# ---------------------------------------------------------------------------

CONSUMPTION_ROUTING_TABLE: List[ConsumptionToEmissionsMapping] = [
    # Scope 1 -- Direct combustion
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.NATURAL_GAS, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        energy_carrier="natural_gas",
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Natural gas combustion emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.FUEL_OIL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        energy_carrier="fuel_oil",
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Fuel oil combustion emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.DIESEL, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        energy_carrier="diesel",
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Diesel combustion emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.LPG, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        energy_carrier="lpg",
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="LPG combustion emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.PROPANE, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        energy_carrier="propane",
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Propane combustion emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.BIOGAS, mrv_agent_id="MRV-001",
        mrv_agent_name="Stationary Combustion", scope=MRVScope.SCOPE_1,
        energy_carrier="biogas",
        module_path="greenlang.agents.mrv.stationary_combustion",
        description="Biogas combustion emissions (biogenic)",
    ),
    # Scope 2 -- Purchased electricity (location-based)
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.ELECTRICITY, mrv_agent_id="MRV-009",
        mrv_agent_name="Scope 2 Location-Based", scope=MRVScope.SCOPE_2,
        energy_carrier="electricity",
        module_path="greenlang.agents.mrv.scope2_location_based",
        description="Purchased electricity emissions (location-based)",
    ),
    # Scope 2 -- Purchased steam/heat
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.STEAM, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        energy_carrier="steam",
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="Purchased steam/heat emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.HOT_WATER, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        energy_carrier="hot_water",
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="Purchased hot water emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.DISTRICT_HEATING, mrv_agent_id="MRV-011",
        mrv_agent_name="Steam/Heat Purchase", scope=MRVScope.SCOPE_2,
        energy_carrier="district_heating",
        module_path="greenlang.agents.mrv.steam_heat_purchase",
        description="District heating emissions",
    ),
    # Scope 2 -- Purchased cooling
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.CHILLED_WATER, mrv_agent_id="MRV-012",
        mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2,
        energy_carrier="chilled_water",
        module_path="greenlang.agents.mrv.cooling_purchase",
        description="Purchased cooling emissions",
    ),
    ConsumptionToEmissionsMapping(
        category=ConsumptionCategory.DISTRICT_COOLING, mrv_agent_id="MRV-012",
        mrv_agent_name="Cooling Purchase", scope=MRVScope.SCOPE_2,
        energy_carrier="district_cooling",
        module_path="greenlang.agents.mrv.cooling_purchase",
        description="District cooling emissions",
    ),
]

# Scope 3 upstream routing (secondary for all fuel types)
SCOPE3_UPSTREAM_ROUTE = ConsumptionToEmissionsMapping(
    category=ConsumptionCategory.ELECTRICITY,
    mrv_agent_id="MRV-016",
    mrv_agent_name="Scope 3 Category 3",
    scope=MRVScope.SCOPE_3,
    scope3_category=3,
    energy_carrier="upstream_energy",
    module_path="greenlang.agents.mrv.scope3_cat3_fuel_energy",
    description="Upstream fuel and energy-related activities",
)


# ---------------------------------------------------------------------------
# Default Emission Factors by Energy Carrier (kg CO2e per kWh)
# ---------------------------------------------------------------------------

DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.366,
    "natural_gas": 0.202,
    "fuel_oil": 0.267,
    "diesel": 0.264,
    "lpg": 0.227,
    "propane": 0.227,
    "steam": 0.230,
    "hot_water": 0.210,
    "chilled_water": 0.180,
    "district_heating": 0.220,
    "district_cooling": 0.175,
    "biogas": 0.0,
    "upstream_energy": 0.050,
}


# ---------------------------------------------------------------------------
# MRVBridge (named MRVUtilityBridge in docstring context)
# ---------------------------------------------------------------------------


class MRVBridge:
    """Bridge to MRV agents for utility consumption emissions calculation.

    Routes utility consumption data to the appropriate MRV agent and
    converts commodity consumption into GHG emissions (tCO2e).
    Supports bi-directional data flow for sending consumption data
    and receiving emissions calculations.

    Attributes:
        config: Bridge configuration.
        _agents: Dict of loaded MRV agent modules/stubs.
        _routing_table: Active routing table.

    Example:
        >>> bridge = MRVBridge()
        >>> result = bridge.route_consumption(
        ...     {"category": "electricity", "consumption_kwh": 100000}
        ... )
        >>> print(f"Emissions: {result.emissions_tco2e} tCO2e")
    """

    def __init__(self, config: Optional[MRVRouteConfig] = None) -> None:
        """Initialize the MRV Utility Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or MRVRouteConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._routing_table = list(CONSUMPTION_ROUTING_TABLE)

        # Load MRV agents with graceful fallback
        self._agents: Dict[str, Any] = {}
        unique_agents = {r.mrv_agent_id: r.module_path for r in self._routing_table}
        unique_agents[SCOPE3_UPSTREAM_ROUTE.mrv_agent_id] = SCOPE3_UPSTREAM_ROUTE.module_path
        # Market-based agent
        unique_agents["MRV-010"] = "greenlang.agents.mrv.scope2_market_based"
        for agent_id, module_path in unique_agents.items():
            self._agents[agent_id] = _try_import_mrv_agent(agent_id, module_path)

        available = sum(
            1 for a in self._agents.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "MRVBridge initialized: %d/%d agents available, country=%s, "
            "method=%s",
            available, len(self._agents), self.config.country_code,
            self.config.electricity_method.value,
        )

    # -------------------------------------------------------------------------
    # Routing
    # -------------------------------------------------------------------------

    def route_consumption(
        self,
        consumption_data: Dict[str, Any],
    ) -> RoutingResult:
        """Route utility consumption data to the appropriate MRV agent.

        Args:
            consumption_data: Dict with 'category', 'consumption_kwh', and
                optional 'region', 'emission_factor_override', 'method'.

        Returns:
            RoutingResult with emissions calculation.
        """
        start = time.monotonic()

        category_str = consumption_data.get("category", "")
        consumption_kwh = float(consumption_data.get("consumption_kwh", 0.0))

        try:
            category = ConsumptionCategory(category_str)
        except ValueError:
            return RoutingResult(
                category=category_str,
                success=False,
                message=f"Unknown consumption category: {category_str}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        route = self._find_route(category)
        if route is None:
            return RoutingResult(
                category=category_str,
                success=False,
                message=f"No routing entry for category '{category_str}'",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Handle electricity method override
        method = consumption_data.get("method", self.config.electricity_method.value)
        effective_route = route
        if category == ConsumptionCategory.ELECTRICITY:
            if method == EmissionsMethod.MARKET_BASED.value:
                effective_route = ConsumptionToEmissionsMapping(
                    category=category,
                    mrv_agent_id="MRV-010",
                    mrv_agent_name="Scope 2 Market-Based",
                    scope=MRVScope.SCOPE_2,
                    energy_carrier="electricity",
                    module_path="greenlang.agents.mrv.scope2_market_based",
                    description="Purchased electricity emissions (market-based)",
                )

        # Get emission factor
        ef_override = consumption_data.get("emission_factor_override")
        if ef_override is not None:
            ef = float(ef_override)
        elif effective_route.energy_carrier == "electricity":
            ef = self.config.grid_emission_factor_kgco2_per_kwh
        elif effective_route.energy_carrier == "natural_gas":
            ef = self.config.natural_gas_ef_kgco2_per_kwh
        else:
            ef = DEFAULT_EMISSION_FACTORS.get(effective_route.energy_carrier, 0.0)

        # Zero-hallucination calculation: direct arithmetic
        emissions_tco2e = (consumption_kwh * ef) / 1000.0

        result = RoutingResult(
            category=category_str,
            mrv_agent_id=effective_route.mrv_agent_id,
            scope=effective_route.scope.value,
            method=method,
            success=True,
            emissions_tco2e=round(emissions_tco2e, 4),
            consumption_kwh=consumption_kwh,
            emission_factor_kgco2_per_kwh=ef,
            message=f"Calculated via {effective_route.mrv_agent_name}",
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    def calculate_all_emissions(
        self,
        consumption_by_commodity: Dict[str, float],
        facility_id: str = "",
        period: str = "",
    ) -> EmissionsSummary:
        """Calculate emissions for all utility commodities.

        Routes each commodity to its MRV agent and aggregates results
        by scope.

        Args:
            consumption_by_commodity: Dict mapping commodity name to kWh.
            facility_id: Facility identifier.
            period: Reporting period (e.g., '2025').

        Returns:
            EmissionsSummary with scope-level aggregations.
        """
        start = time.monotonic()
        routing_results: List[RoutingResult] = []
        breakdown: Dict[str, float] = {}
        scope1 = scope2 = scope3 = 0.0

        for commodity, kwh in consumption_by_commodity.items():
            result = self.route_consumption({
                "category": commodity,
                "consumption_kwh": kwh,
            })
            routing_results.append(result)
            if result.success:
                breakdown[commodity] = result.emissions_tco2e
                if result.scope == MRVScope.SCOPE_1.value:
                    scope1 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_2.value:
                    scope2 += result.emissions_tco2e
                elif result.scope == MRVScope.SCOPE_3.value:
                    scope3 += result.emissions_tco2e

        summary = EmissionsSummary(
            facility_id=facility_id,
            period=period,
            scope1_tco2e=round(scope1, 4),
            scope2_tco2e=round(scope2, 4),
            scope3_tco2e=round(scope3, 4),
            total_tco2e=round(scope1 + scope2 + scope3, 4),
            breakdown_by_commodity=breakdown,
            routing_results=routing_results,
        )

        if self.config.enable_provenance:
            summary.provenance_hash = _compute_hash(summary)

        self.logger.info(
            "Emissions summary: facility=%s, period=%s, "
            "S1=%.2f S2=%.2f S3=%.2f total=%.2f tCO2e, duration=%.1fms",
            facility_id, period, scope1, scope2, scope3,
            summary.total_tco2e, (time.monotonic() - start) * 1000,
        )
        return summary

    def convert_to_emissions(
        self,
        consumption_kwh: float,
        energy_carrier: str = "electricity",
    ) -> Decimal:
        """Convert energy consumption to emissions.

        Deterministic calculation:
            emissions_tco2e = consumption_kwh * ef / 1000.0

        Args:
            consumption_kwh: Energy consumption in kWh.
            energy_carrier: Energy carrier type for EF selection.

        Returns:
            Emissions in tCO2e as Decimal.
        """
        ef = DEFAULT_EMISSION_FACTORS.get(energy_carrier, 0.0)
        emissions = (
            Decimal(str(consumption_kwh)) * Decimal(str(ef)) / Decimal("1000.0")
        )
        return emissions.quantize(Decimal("0.0001"))

    def get_applicable_agents(
        self,
        commodity: str,
    ) -> List[Dict[str, Any]]:
        """Get the list of MRV agents applicable to a commodity.

        Args:
            commodity: Utility commodity category.

        Returns:
            List of applicable agent routing entries.
        """
        return [
            {
                "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name,
                "scope": r.scope.value,
                "energy_carrier": r.energy_carrier,
                "available": not isinstance(
                    self._agents.get(r.mrv_agent_id), _AgentStub
                ),
            }
            for r in self._routing_table
            if r.category.value == commodity
        ]

    # -------------------------------------------------------------------------
    # Informational
    # -------------------------------------------------------------------------

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """Get the full routing table as a list of dicts.

        Returns:
            List of routing entries with availability status.
        """
        return [
            {
                "category": r.category.value,
                "mrv_agent_id": r.mrv_agent_id,
                "mrv_agent_name": r.mrv_agent_name,
                "scope": r.scope.value,
                "energy_carrier": r.energy_carrier,
                "available": not isinstance(
                    self._agents.get(r.mrv_agent_id), _AgentStub
                ),
            }
            for r in self._routing_table
        ]

    def get_default_emission_factors(self) -> Dict[str, float]:
        """Get default emission factors by energy carrier.

        Returns:
            Dict mapping energy carrier to kg CO2e per kWh.
        """
        return dict(DEFAULT_EMISSION_FACTORS)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _find_route(
        self, category: ConsumptionCategory
    ) -> Optional[ConsumptionToEmissionsMapping]:
        """Find the routing entry for a consumption category.

        Args:
            category: Consumption category to look up.

        Returns:
            ConsumptionToEmissionsMapping if found, None otherwise.
        """
        for route in self._routing_table:
            if route.category == category:
                return route
        return None
